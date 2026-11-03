"""Fused linear + cross-entropy for the tied large-vocabulary head.

    loss_t = logsumexp(h_t E^T + b) - (h_t E_{y_t} + b_{y_t})

The (tokens, vocab) logits are never stored. The forward pass runs over chunks of tokens and keeps only the
log-sum-exp per token. The backward pass recomputes each chunk's logits, forms softmax - onehot in place and
accumulates dE, db and writes dh. Matmuls take `dtype` operands with fp32 accumulation; log-sum-exp is fp32.

Use (h is the head input, e.g. rms(o) * out_g):
    loss_sum, count = fused_ce(h, params["emb"], params["head_b"], y, m)
    per_tok = fused_ce(h, emb, b, y, return_per_token=True)   # (..., ) fp32, for evaluation
    per_tok = fused_ce(h, emb, b, y, return_per_token=True, impl="pallas")   # Triton kernels, GPU only

    python models/yggdrasil/training/head.py check     # loss and gradients against the naive form
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _logits(h, e, b, dtype):
    return jnp.matmul(h.astype(dtype), e.astype(dtype).T, preferred_element_type=jnp.float32) + b


def _chunks(x, chunk):
    return x.reshape((-1, chunk) + x.shape[1:])


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _ce_rows(h, e, b, y, chunk, dtype):
    return _ce_fwd(h, e, b, y, chunk, dtype)[0]


def _ce_fwd(h, e, b, y, chunk, dtype):
    def one(args):
        hc, yc = args
        z = _logits(hc, e, b, dtype)
        lse = jax.nn.logsumexp(z, -1)
        return lse - jnp.take_along_axis(z, yc[:, None], -1)[:, 0], lse

    loss, lse = jax.lax.map(one, (_chunks(h, chunk), _chunks(y, chunk)))
    return loss.reshape(-1), (h, e, b, y, lse)


def _ce_bwd(chunk, dtype, res, g):
    h, e, b, y, lse = res
    ed = e.astype(dtype)

    def body(acc, args):
        hc, yc, lc, gc = args
        z = _logits(hc, e, b, dtype)
        d = jnp.exp(z - lc[:, None])
        d = d.at[jnp.arange(chunk), yc].add(-1.0) * gc[:, None]  # (softmax - onehot) * dloss
        db = d.sum(0)
        d = d.astype(dtype)
        dh = jnp.matmul(d, ed, preferred_element_type=jnp.float32)
        de = jnp.matmul(d.T, hc.astype(dtype), preferred_element_type=jnp.float32)
        return (acc[0] + de, acc[1] + db), dh

    zero = (jnp.zeros(e.shape, jnp.float32), jnp.zeros(b.shape, jnp.float32))
    (de, db), dh = jax.lax.scan(body, zero, (_chunks(h, chunk), _chunks(y, chunk), lse, _chunks(g, chunk)))
    return dh.reshape(h.shape).astype(h.dtype), de.astype(e.dtype), db.astype(b.dtype), None


_ce_rows.defvjp(_ce_fwd, _ce_bwd)


# ------------------------------------------------------------------------------------------------- Pallas (Triton)
# Same loss, two kernels; no (tokens, vocab) array is ever written to HBM.
#   forward, one program per token block: stream vocab tiles of E, online log-sum-exp, and (when a gradient is
#     needed) the softmax-weighted sum pe = softmax(z) E, rescaled like flash attention. Then dh = g (pe - E_y).
#   backward, one program per vocab tile: stream token blocks, recompute z, p = exp(z - lse) g, accumulate
#     dE_tile += p^T h and db_tile += sum p in registers. No atomics. The label terms (-g h into dE_y, -g into db_y,
#     the label logit) are gathers and scatters of size (tokens, d) done in JAX.
# h and E are padded to the tile sizes: d to a multiple of bk, vocab to bv (bias -1e30), tokens to bt.
# For wide d (768) the pe and dE columns are split over a second grid axis; each slice recomputes the logits.

_NEG = -1e30


def _tiles(d, dtype=jnp.bfloat16):
    """(bt, bv, bk, warps, stages, slices) for the forward and the backward kernel, per embedding width.
    slices > 1 splits the d columns of pe and dE over a second grid axis (each slice recomputes the logits)."""
    if jnp.dtype(dtype).itemsize > 2:  # fp32 operands (checks only): small tiles fit shared memory
        return ((32, 32, 32, 4, 1, 1),) * 2 if d <= 256 else ((16, 16, 32, 4, 1, 1),) * 2
    if d <= 256:  # tuned on L40S and RTX PRO 5000 (101 KB shared memory per block)
        return (64, 64, 32, 4, 2, 1), (64, 64, 32, 4, 2, 1)
    if d <= 512:
        return (64, 64, 64, 4, 2, 1), (32, 64, 64, 4, 2, 1)
    return (32, 64, 64, 4, 1, 2), (32, 32, 64, 4, 1, 2)  # d 768: one stage fits shared memory; about the chunked speed


def _prec(dtype):
    return jax.lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else None


def _fwd_kernel(h_ref, e_ref, b_ref, lse_ref, *pe_ref, bv, bk, nk, nv, ns, prec, wide):
    # wide: load every chunk right before its dot, so shared memory holds one chunk, not a full row
    bt = h_ref.shape[0]
    hk = None if wide else [h_ref[:, pl.ds(k * bk, bk)] for k in range(nk)]
    nks = nk // ns
    s0 = pl.program_id(1) * (nks * bk)
    m = jnp.full((bt,), _NEG, jnp.float32)
    s = jnp.zeros((bt,), jnp.float32)
    acc = [jnp.zeros((bt, bk), jnp.float32) for _ in range(nks)] if pe_ref else []

    def body(j, carry):
        m, s, acc = carry
        rows = pl.ds(j * bv, bv)
        z = b_ref[rows][None, :]
        ek = []
        for k in range(nk):
            e = e_ref[rows, pl.ds(k * bk, bk)]
            ek.append(e)
            z = z + plgpu.dot(h_ref[:, pl.ds(k * bk, bk)] if wide else hk[k], e, trans_b=True, precision=prec)
        m2 = jnp.maximum(m, z.max(1))
        c = jnp.exp(m - m2)
        p = jnp.exp(z - m2[:, None])
        s = s * c + p.sum(1)
        if acc:
            p = p.astype(ek[0].dtype)
            acc = [a * c[:, None] + plgpu.dot(
                p, e_ref[rows, pl.ds(s0 + k * bk, bk)] if wide else ek[k], precision=prec)
                for k, a in enumerate(acc)]
        return m2, s, acc

    m, s, acc = jax.lax.fori_loop(0, nv, body, (m, s, acc))
    lse_ref[...] = m + jnp.log(s)  # every slice writes the same values
    for k, a in enumerate(acc):
        pe_ref[0][:, pl.ds(k * bk, bk)] = a / s[:, None]


def _bwd_kernel(e_ref, b_ref, h_ref, lse_ref, g_ref, de_ref, db_ref, *, bt, bk, nk, nt, ns, prec, wide):
    bv = e_ref.shape[0]
    ek = None if wide else [e_ref[:, pl.ds(k * bk, bk)] for k in range(nk)]
    nks = nk // ns
    s0 = pl.program_id(1) * (nks * bk)
    bias = b_ref[...]

    def body(i, carry):
        acc, db = carry
        rows = pl.ds(i * bt, bt)
        z = bias[:, None]  # (bv, bt): z transposed, so p^T h needs no transpose
        hk = []
        for k in range(nk):
            h = h_ref[rows, pl.ds(k * bk, bk)]
            hk.append(h)
            z = z + plgpu.dot(e_ref[:, pl.ds(k * bk, bk)] if wide else ek[k], h, trans_b=True, precision=prec)
        p = jnp.exp(z - lse_ref[rows][None, :]) * g_ref[rows][None, :]
        db = db + p.sum(1)
        p = p.astype(hk[0].dtype)
        return [a + plgpu.dot(p, h_ref[rows, pl.ds(s0 + k * bk, bk)] if wide else hk[k], precision=prec)
                for k, a in enumerate(acc)], db

    acc = [jnp.zeros((bv, bk), jnp.float32) for _ in range(nks)]
    acc, db = jax.lax.fori_loop(0, nt, body, (acc, jnp.zeros((bv,), jnp.float32)))
    for k, a in enumerate(acc):
        de_ref[:, pl.ds(k * bk, bk)] = a
    db_ref[...] = db  # every slice writes the same values


def _pad_to(x, axis, mult, value=0):
    pad = -x.shape[axis] % mult
    if not pad:
        return x
    w = [(0, 0)] * x.ndim
    w[axis] = (0, pad)
    return jnp.pad(x, w, constant_values=value)


def _wide(dp, ns, dtype):
    return ns > 1 or dp * jnp.dtype(dtype).itemsize > 1024


def _pl_fwd_call(h, e, b, dtype, with_pe, tiles):
    """h (n, d), e (V, d), b (V,) -> lse (n,) and pe (n, d) fp32 or None. n is a multiple of bt."""
    (bt, bv, bk, warps, stages, ns), _ = tiles
    ns = ns if with_pe else 1
    n, d = h.shape
    hp = _pad_to(h.astype(dtype), 1, bk * ns)
    ep = _pad_to(_pad_to(e.astype(dtype), 1, bk * ns), 0, bv)
    bp = _pad_to(b.astype(jnp.float32), 0, bv, _NEG)
    dp, vp = hp.shape[1], ep.shape[0]
    out_shape = [jax.ShapeDtypeStruct((n,), jnp.float32)]
    out_specs = [pl.BlockSpec((bt,), lambda i, s: (i,))]
    if with_pe:
        out_shape.append(jax.ShapeDtypeStruct((n, dp), jnp.float32))
        out_specs.append(pl.BlockSpec((bt, dp // ns), lambda i, s: (i, s)))
    kern = functools.partial(_fwd_kernel, bv=bv, bk=bk, nk=dp // bk, nv=vp // bv, ns=ns, prec=_prec(dtype),
                             wide=_wide(dp, ns, dtype))
    out = pl.pallas_call(
        kern, grid=(n // bt, ns), out_shape=out_shape, out_specs=out_specs,
        in_specs=[pl.BlockSpec((bt, dp), lambda i, s: (i, 0)), pl.BlockSpec((vp, dp), lambda i, s: (0, 0)),
                  pl.BlockSpec((vp,), lambda i, s: (0,))],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages), name="ce_fwd",
    )(hp, ep, bp)
    return out[0], (out[1][:, :d] if with_pe else None)


def _pl_bwd_call(h, e, b, lse, g, dtype, tiles):
    """-> dE (V, d) and db (V,) fp32, softmax part only."""
    _, (bt, bv, bk, warps, stages, ns) = tiles
    n, d = h.shape
    v = e.shape[0]
    hp = _pad_to(_pad_to(h.astype(dtype), 1, bk * ns), 0, bt)
    lp, gp = _pad_to(lse, 0, bt), _pad_to(g.astype(jnp.float32), 0, bt)
    ep = _pad_to(_pad_to(e.astype(dtype), 1, bk * ns), 0, bv)
    bp = _pad_to(b.astype(jnp.float32), 0, bv, _NEG)
    (np_, dp), vp = hp.shape, ep.shape[0]
    kern = functools.partial(_bwd_kernel, bt=bt, bk=bk, nk=dp // bk, nt=np_ // bt, ns=ns, prec=_prec(dtype),
                             wide=_wide(dp, ns, dtype))
    de, db = pl.pallas_call(
        kern, grid=(vp // bv, ns),
        out_shape=[jax.ShapeDtypeStruct((vp, dp), jnp.float32), jax.ShapeDtypeStruct((vp,), jnp.float32)],
        in_specs=[pl.BlockSpec((bv, dp), lambda j, s: (j, 0)), pl.BlockSpec((bv,), lambda j, s: (j,)),
                  pl.BlockSpec((np_, dp), lambda j, s: (0, 0)), pl.BlockSpec((np_,), lambda j, s: (0,)),
                  pl.BlockSpec((np_,), lambda j, s: (0,))],
        out_specs=[pl.BlockSpec((bv, dp // ns), lambda j, s: (j, s)), pl.BlockSpec((bv,), lambda j, s: (j,))],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages), name="ce_bwd",
    )(ep, bp, hp, lp, gp)
    return de[:v, :d], db[:v]


def _label_logit(h, e, b, y, dtype):
    hd = h.astype(dtype).astype(jnp.float32)
    return (hd * e[y].astype(dtype).astype(jnp.float32)).sum(-1) + b[y]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _pl_rows(h, e, b, y, dtype, tiles):
    lse, _ = _pl_fwd_call(h, e, b, dtype, False, tiles)
    return lse - _label_logit(h, e, b, y, dtype)


def _pl_fwd(h, e, b, y, dtype, tiles):
    lse, pe = _pl_fwd_call(h, e, b, dtype, True, tiles)
    return lse - _label_logit(h, e, b, y, dtype), (h, e, b, y, lse, pe)


def _pl_bwd(dtype, tiles, res, g):
    h, e, b, y, lse, pe = res
    de, db = _pl_bwd_call(h, e, b, lse, g, dtype, tiles)
    hd = h.astype(dtype).astype(jnp.float32)
    dh = g[:, None] * (pe - e[y].astype(dtype).astype(jnp.float32))
    de = de.at[y].add(-g[:, None] * hd)
    db = db.at[y].add(-g)
    return dh.astype(h.dtype), de.astype(e.dtype), db.astype(b.dtype), None


_pl_rows.defvjp(_pl_fwd, _pl_bwd)


def fused_ce(h, emb, bias, labels, mask=None, chunk=2048, dtype=jnp.bfloat16, return_per_token=False,
             impl="chunk", tiles=None):
    """h (..., d), emb (V, d), bias (V,) or None, labels (...) int -> (masked loss sum, mask sum), both fp32 scalars.
    return_per_token: the per-token loss (...) instead. Gradients flow to h, emb and bias.
    impl: "chunk" (XLA, chunked logits) or "pallas" (Triton kernels, GPU only; chunk is unused, tiles from _tiles)."""
    shape = labels.shape
    h2, y = h.reshape(-1, h.shape[-1]), labels.reshape(-1).astype(jnp.int32)
    b = jnp.zeros(emb.shape[:1], jnp.float32) if bias is None else bias
    n = y.shape[0]
    if impl == "pallas":
        tiles = tiles or _tiles(h.shape[-1], dtype)
        chunk = tiles[0][0]  # pad tokens to the forward token block
    else:
        chunk = min(chunk, n)
    pad = -n % chunk
    if pad:
        h2, y = jnp.pad(h2, ((0, pad), (0, 0))), jnp.pad(y, (0, pad))
    if impl == "pallas":
        loss = _pl_rows(h2, emb, b, y, dtype, tiles)[:n].reshape(shape)
    else:
        loss = _ce_rows(h2, emb, b, y, chunk, dtype)[:n].reshape(shape)
    if return_per_token:
        return loss
    m = jnp.ones(shape, jnp.float32) if mask is None else mask.astype(jnp.float32)
    return (loss * m).sum(), m.sum()


def naive_ce(h, emb, bias, labels, mask=None, dtype=jnp.bfloat16):
    """Reference: full logits and log_softmax."""
    logp = jax.nn.log_softmax(_logits(h, emb, bias, dtype), -1)
    loss = -jnp.take_along_axis(logp, labels[..., None], -1)[..., 0]
    m = jnp.ones(labels.shape, jnp.float32) if mask is None else mask.astype(jnp.float32)
    return (loss * m).sum(), m.sum()


# ----------------------------------------------------------------------------------------------------------- check

def _inputs(n, d, v, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    o = jax.random.normal(k[0], (n, d))
    h = o * jax.lax.rsqrt(jnp.mean(o * o, -1, keepdims=True) + 1e-6)
    e = 0.02 * jax.random.normal(k[1], (v, d))
    b = 0.1 * jax.random.normal(k[2], (v,))
    y = jax.random.randint(k[3], (n,), 0, v)
    m = (jax.random.uniform(k[4], (n,)) > 0.2).astype(jnp.float32)
    return h, e, b, y, m


def check():
    with jax.default_matmul_precision("highest"):  # no TF32 in the fp32 reference on GPU
        _check()


def _check():
    for dtype, tol in [(jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)]:
        for n, d, v, chunk in [(1000, 160, 50257, 256), (4096, 384, 50257, 1024), (37, 64, 1000, 16)]:
            h, e, b, y, m = _inputs(n, d, v)
            mean = lambda f: (lambda h, e, b: (lambda s, c: s / c)(*f(h, e, b)))
            fn = mean(lambda h, e, b: fused_ce(h, e, b, y, m, chunk, dtype))
            nv = mean(lambda h, e, b: naive_ce(h, e, b, y, m, jnp.float32))  # reference in fp32
            lf, gf = jax.jit(jax.value_and_grad(fn, (0, 1, 2)))(h, e, b)
            ln, gn = jax.jit(jax.value_and_grad(nv, (0, 1, 2)))(h, e, b)
            rel = [float(jnp.linalg.norm(a - c) / jnp.linalg.norm(c)) for a, c in zip(gf, gn)]
            pt = fused_ce(h, e, b, y, None, chunk, dtype, return_per_token=True)
            ok = abs(float(lf - ln)) < tol * abs(float(ln)) and max(rel) < tol and \
                abs(float((pt * m).sum() / m.sum() - lf)) < 1e-5 * abs(float(lf))
            print(f"{jnp.dtype(dtype).name:8s} n {n:5d} d {d} V {v} chunk {chunk}: loss {float(lf):.6f} naive "
                  f"{float(ln):.6f}  rel grad err h {rel[0]:.1e} E {rel[1]:.1e} b {rel[2]:.1e}  {'ok' if ok else 'FAIL'}",
                  flush=True)
            assert ok


if __name__ == "__main__":
    check()
