"""Routed leaf matmuls without padding (Pallas, Triton backend), in the manner of MegaBlocks.

Each token t reads K leaves l_tq with straight-through weights w_tq whose value is one-hot on slot 0, so

    y_t = x_t W[l_t0]^T + b[l_t0]      (value: slot 0 only)
    dw_tq = <dy_t, x_t W[l_tq]^T + b[l_tq]>,  dx_t = dy_t W[l_t0],  dW[g] = sum_{t: l_t0 = g} dy_t^T x_t.

Rows are sorted by leaf once per branch (plan). The sorted rows are cut into tiles of TM rows, and a tile that holds
rows of several leaves is visited once per leaf with the other rows masked, so no row is padded or dropped. The kernels
gather token rows by index and scatter results back in token order, so no sorted copy of x or dy is stored. The
slot weights' gradient needs no (rows, m) output: each visit reduces its tile against dy to one number per row; all
slots go through the same kernel, so leaves with equal tables give equal slot gradients. dW splits a leaf's rows into
chunks of CH rows, so a leaf that holds most tokens does not run on a few programs alone. db comes from the same read
of dy that rounds it to bf16. Several (x, W) pairs with one output run as one kernel, without a concat.
Operands are bf16, accumulation fp32.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

ON = True  # train.py uses this module on GPU
TM, TM_ALL = 64, 128  # rows per tile: slot 0 plan, all-slot plan (slot weight gradient only)
CH = 2048  # dW: rows per chunk; a leaf with more rows is split into chunks whose partial sums are added after
CFG = {  # TN, TK, warps, stages (mm: forward, dx: x gradient, dot: slot weights); dw: TN, TK, rows per step, warps, stages
    "mm": (64, 64, 4, 3), "dx": (64, 64, 4, 3), "dot": (128, 64, 4, 3), "dw": (128, 128, 32, 8, 2),
    # tuned on one L40S at 32 x 1024 tokens, keyed (kind, k, m)
    ("mm", 512, 2048): (128, 64, 4, 3), ("mm", 2048, 256): (64, 128, 4, 4),
    ("dx", 256, 2048): (128, 128, 8, 3), ("dx", 2048, 256): (256, 32, 8, 3),
    ("dot", 512, 2048): (256, 32, 8, 3), ("dot", 2048, 256): (256, 64, 8, 3),
    ("dw", 256, 2048): (128, 256, 32, 8, 2), ("dw", 2048, 256): (128, 128, 64, 8, 2),
    ("mm", 128, 1024): (256, 32, 8, 3), ("mm", 1024, 64): (64, 128, 4, 2),
    ("dx", 64, 1024): (256, 128, 4, 2), ("dx", 1024, 64): (256, 32, 8, 2),
    ("dot", 128, 1024): (128, 32, 4, 2), ("dot", 1024, 64): (64, 64, 4, 3),
    ("dw", 64, 1024): (256, 128, 32, 4, 2), ("dw", 1024, 64): (128, 256, 32, 4, 2)}


CFG_F32 = {"mm": (64, 32, 4, 2), "dx": (64, 32, 4, 2), "dot": (64, 32, 4, 2), "dw": (64, 64, 32, 4, 2)}  # 4-byte operands


def _cfg(kind, k, m, dtype):
    """Per-shape entries (kind, k, m) override the default of each kind; they are tuned for 2-byte operands."""
    return CFG.get((kind, k, m), CFG[kind]) if jnp.dtype(dtype).itemsize <= 2 else CFG_F32[kind]


def _visits(start, end, n_tiles, tm):
    """(tile, group) pairs: every tile that holds rows of a group, padded to n_tiles + G with group -1."""
    G = start.shape[0]
    first = start // tm
    last = jnp.where(end > start, (end - 1) // tm, first - 1)
    n = jnp.maximum(last - first + 1, 0)
    off = jnp.cumsum(n) - n
    v = jnp.arange(n_tiles + G)
    grp = jnp.clip(jnp.searchsorted(off + n, v, side="right"), 0, G - 1)
    tile = jnp.clip(first[grp] + v - off[grp], 0, n_tiles - 1)
    return tile.astype(jnp.int32), jnp.where(v < (off + n)[-1], grp, -1).astype(jnp.int32)


def _chunks(start, end, R):
    """dW visits: (group, chunk) for every group (one chunk at least, so empty groups write zeros) and every further CH
    rows, padded to G + R // CH with group -1. Chunk c >= 1 of group g is extra number v - g - 1 (owner g)."""
    G = start.shape[0]
    n = jnp.maximum((end - start + CH - 1) // CH, 1)
    off = jnp.cumsum(n) - n
    v = jnp.arange(G + R // CH)
    grp = jnp.clip(jnp.searchsorted(off + n, v, side="right"), 0, G - 1)
    c = v - off[grp]
    ok = v < (off + n)[-1]
    owner = jnp.full((max(R // CH, 1),), G, jnp.int32).at[jnp.where(ok & (c > 0), v - grp - 1, R)].set(
        grp.astype(jnp.int32), mode="drop")
    return jnp.where(ok, grp, -1).astype(jnp.int32), c.astype(jnp.int32), owner


def _segments(ids, G, scale=1, tm=TM):
    """ids (R,) -> sorted positions' token index (R padded to tm), start, end, visits of tm-row tiles."""
    R = ids.shape[0]
    order = jnp.argsort(ids).astype(jnp.int32)
    count = jnp.bincount(ids, length=G).astype(jnp.int32)
    end = jnp.cumsum(count).astype(jnp.int32)
    start = end - count
    pad = -R % tm
    tok = jnp.concatenate([order // scale, jnp.zeros((pad,), jnp.int32)])
    tile, grp = _visits(start, end, (R + pad) // tm, tm)
    return dict(ids=ids, order=order, tok=tok, start=start, end=end, tile=tile, grp=grp)


def plan(leaf, G):
    """leaf (N, K) -> sort plans for slot 0 (values, dx, dW) and for all slots (slot weight gradient)."""
    N, K = leaf.shape
    leaf = jax.lax.stop_gradient(leaf).astype(jnp.int32)
    s0 = _segments(leaf[:, 0], G)
    s0["cgrp"], s0["chunk"], s0["owner"] = _chunks(s0["start"], s0["end"], N)
    return dict(s0=s0, all=_segments(leaf.reshape(-1), G, K, TM_ALL))


def _full(a):
    return pl.BlockSpec(a.shape, lambda *_: (0,) * a.ndim)


def _rows(tile_ref, grp_ref, start_ref, end_ref, tok_ref, v, tm=TM):
    gr = grp_ref[v]
    g = jnp.maximum(gr, 0)
    r0 = pl.multiple_of(tile_ref[v] * tm, tm)
    pos = r0 + jnp.arange(tm)
    keep = (pos >= start_ref[g]) & (pos < end_ref[g]) & (gr >= 0)
    return gr, g, r0, keep, tok_ref[pl.ds(r0, tm)]


def _matmul(tok, g, j, pairs, tn, tk):
    """sum_i x_i[tok] W_i[g]^T over the j-th block of tn output columns, fp32; the k loops are software-pipelined."""
    acc = jnp.zeros((tok.shape[0], tn), jnp.float32)
    for x_ref, w_ref in pairs:
        def body(kk, acc, x_ref=x_ref, w_ref=w_ref):
            c = pl.multiple_of(kk * tk, tk)
            return acc + jnp.dot(x_ref[tok, pl.ds(c, tk)], w_ref[g, pl.ds(j * tn, tn), pl.ds(c, tk)].T,
                                 preferred_element_type=jnp.float32)
        acc = jax.lax.fori_loop(0, x_ref.shape[1] // tk, body, acc)
    return acc


def _pairs_args(pln, xs, ws, b):
    return [pln["tile"], pln["grp"], pln["start"], pln["end"], pln["tok"]] + \
        [a for xw in zip(xs, ws) for a in xw] + ([b] if b is not None else [])


def _mm_kernel(tile_ref, grp_ref, start_ref, end_ref, tok_ref, *rest, n, tn, tk, bias):
    """y[tok] = sum_i x_i[tok] W_i[g]^T (+ b[g]) for the rows of group g in one tile."""
    pairs = [rest[2 * i:2 * i + 2] for i in range(n)]
    b_ref, y_ref = rest[2 * n:] if bias else (None, rest[2 * n])
    j, v = pl.program_id(0), pl.program_id(1)
    gr, g, r0, keep, tok = _rows(tile_ref, grp_ref, start_ref, end_ref, tok_ref, v)

    @pl.when(gr >= 0)
    def _():
        acc = _matmul(tok, g, j, pairs, tn, tk)
        if bias:
            acc += b_ref[g, pl.ds(j * tn, tn)][None, :]
        plgpu.store(y_ref.at[tok, pl.ds(j * tn, tn)], acc.astype(y_ref.dtype), mask=keep[:, None])


def _mm(pln, xs, ws, b, N):
    """Token-ordered x_i (N, k_i), w_i (G, m, k_i) -> (N, m) fp32, y_t = sum_i x_it W_i[g_t]^T + b[g_t]."""
    k, m = sum(x.shape[1] for x in xs), ws[0].shape[1]
    tn, tk, warps, stages = _cfg("mm", k, m, xs[0].dtype)
    tn, tk = min(tn, m), min([tk] + [x.shape[1] for x in xs])
    args = _pairs_args(pln, xs, ws, b)
    return pl.pallas_call(
        functools.partial(_mm_kernel, n=len(xs), tn=tn, tk=tk, bias=b is not None),
        grid=(m // tn, pln["tile"].shape[0]), in_specs=[_full(a) for a in args],
        out_specs=pl.BlockSpec((N, m), lambda j, v: (0, 0)), out_shape=jax.ShapeDtypeStruct((N, m), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages))(*args)


def _dx_kernel(tile_ref, grp_ref, start_ref, end_ref, tok_ref, dy_ref, w_ref, dx_ref, *, m, tn, tk):
    """dx[tok] = dy[tok] W[g] over one block of k columns."""
    j, v = pl.program_id(0), pl.program_id(1)
    gr, g, r0, keep, tok = _rows(tile_ref, grp_ref, start_ref, end_ref, tok_ref, v)

    @pl.when(gr >= 0)
    def _():
        def body(kk, acc):
            o = pl.multiple_of(kk * tk, tk)
            return acc + jnp.dot(dy_ref[tok, pl.ds(o, tk)], w_ref[g, pl.ds(o, tk), pl.ds(j * tn, tn)],
                                 preferred_element_type=jnp.float32)
        acc = jax.lax.fori_loop(0, m // tk, body, jnp.zeros((TM, tn), jnp.float32))
        plgpu.store(dx_ref.at[tok, pl.ds(j * tn, tn)], acc.astype(dx_ref.dtype), mask=keep[:, None])


def _dx(pln, dy, w, N):
    """dy (N, m), w (G, m, k) -> dx (N, k) fp32, dx_t = dy_t W[g_t]."""
    m, k = w.shape[1], w.shape[2]
    tn, tk, warps, stages = _cfg("dx", k, m, dy.dtype)
    tn, tk = min(tn, k), min(tk, m)
    args = [pln["tile"], pln["grp"], pln["start"], pln["end"], pln["tok"], dy, w]
    return pl.pallas_call(
        functools.partial(_dx_kernel, m=m, tn=tn, tk=tk),
        grid=(k // tn, pln["tile"].shape[0]), in_specs=[_full(a) for a in args],
        out_specs=pl.BlockSpec((N, k), lambda j, v: (0, 0)), out_shape=jax.ShapeDtypeStruct((N, k), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages))(*args)


def _dw_kernel(grp_ref, chunk_ref, start_ref, end_ref, tok_ref, x_ref, dy_ref, dw_ref, ex_ref, *, tn, tk, rs):
    """dW[g] (tn, tk) block = sum over one chunk of the group's rows of dy^T x, rs sorted rows per step; chunk 0 writes
    dW[g], chunk c >= 1 writes extra number v - g - 1."""
    v, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    gr, c = grp_ref[v], chunk_ref[v]
    g = jnp.maximum(gr, 0)
    s = start_ref[g] + c * CH
    e = jnp.minimum(end_ref[g], s + CH)

    @pl.when(gr >= 0)
    def _():
        def body(t, acc):
            pos = t * rs + jnp.arange(rs)
            tok = tok_ref[pl.ds(pl.multiple_of(t * rs, rs), rs)]
            xb = plgpu.load(x_ref.at[tok, pl.ds(j * tk, tk)], mask=((pos >= s) & (pos < e))[:, None], other=0)
            return acc + jnp.dot(dy_ref[tok, pl.ds(i * tn, tn)].T, xb, preferred_element_type=jnp.float32)
        t0 = s // rs
        acc = jax.lax.fori_loop(t0, jnp.maximum((e + rs - 1) // rs, t0), body, jnp.zeros((tn, tk), jnp.float32))

        @pl.when(c == 0)
        def _():
            plgpu.store(dw_ref.at[g, pl.ds(i * tn, tn), pl.ds(j * tk, tk)], acc)

        @pl.when(c > 0)
        def _():
            plgpu.store(ex_ref.at[v - g - 1, pl.ds(i * tn, tn), pl.ds(j * tk, tk)], acc)


def _dw(pln, x, dy, G):
    """dW (G, m, k) fp32, dW[g] = sum_{t in g} dy_t^T x_t."""
    k, m = x.shape[1], dy.shape[1]
    tn, tk, rs, warps, stages = _cfg("dw", k, m, x.dtype)
    tn, tk = min(tn, m), min(tk, k)
    tok = pln["tok"]
    tok = jnp.concatenate([tok, jnp.zeros((-tok.shape[0] % rs + rs,), jnp.int32)])
    E = pln["owner"].shape[0]
    args = [pln["cgrp"], pln["chunk"], pln["start"], pln["end"], tok, x, dy]
    dw, ex = pl.pallas_call(
        functools.partial(_dw_kernel, tn=tn, tk=tk, rs=rs),
        grid=(pln["cgrp"].shape[0], m // tn, k // tk), in_specs=[_full(a) for a in args],
        out_specs=[pl.BlockSpec((G, m, k), lambda *_: (0, 0, 0)), pl.BlockSpec((E, m, k), lambda *_: (0, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((G, m, k), jnp.float32), jax.ShapeDtypeStruct((E, m, k), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages))(*args)
    return dw.at[pln["owner"]].add(ex, mode="drop")


def _cast_kernel(tile_ref, grp_ref, start_ref, end_ref, tok_ref, dy_ref, out_ref, s_ref, *, tn):
    """out[tok] = dy[tok] in out's dtype for the rows of group g in one tile, and the tile's column sums of dy."""
    j, v = pl.program_id(0), pl.program_id(1)
    gr, g, r0, keep, tok = _rows(tile_ref, grp_ref, start_ref, end_ref, tok_ref, v)
    dyb = plgpu.load(dy_ref.at[tok, pl.ds(j * tn, tn)], mask=keep[:, None], other=0)
    plgpu.store(out_ref.at[tok, pl.ds(j * tn, tn)], dyb.astype(out_ref.dtype), mask=keep[:, None])
    s_ref[v, pl.ds(j * tn, tn)] = jnp.sum(dyb.astype(jnp.float32), axis=0)


def _cast_sum(pln, dy, dtype, G):
    """dy (N, m) -> dy in dtype and db (G, m) fp32 = sum_{t in g} dy_t, from one read of dy."""
    (N, m), V = dy.shape, pln["tile"].shape[0]
    tn = min(256, m)
    args = [pln["tile"], pln["grp"], pln["start"], pln["end"], pln["tok"], dy]
    out, part = pl.pallas_call(
        functools.partial(_cast_kernel, tn=tn), grid=(m // tn, V), in_specs=[_full(a) for a in args],
        out_specs=[pl.BlockSpec((N, m), lambda j, v: (0, 0)), pl.BlockSpec((V, m), lambda j, v: (0, 0))],
        out_shape=[jax.ShapeDtypeStruct((N, m), dtype), jax.ShapeDtypeStruct((V, m), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2))(*args)
    return out, jax.ops.segment_sum(part, jnp.where(pln["grp"] >= 0, pln["grp"], G), G + 1)[:G]


def _dot_kernel(tile_ref, grp_ref, start_ref, end_ref, tok_ref, *rest, n, tn, tk, bias):
    """Partial row dots over one block of m columns: r[j, row] = <dy[tok], sum_i x_i[tok] W_i[g]^T + b[g]>."""
    pairs = [rest[2 * i:2 * i + 2] for i in range(n)]
    dy_ref = rest[2 * n]
    b_ref, r_ref = rest[2 * n + 1:] if bias else (None, rest[2 * n + 1])
    j, v = pl.program_id(0), pl.program_id(1)
    gr, g, r0, keep, tok = _rows(tile_ref, grp_ref, start_ref, end_ref, tok_ref, v, TM_ALL)

    @pl.when(gr >= 0)
    def _():
        acc = _matmul(tok, g, j, pairs, tn, tk)
        if bias:
            acc += b_ref[g, pl.ds(j * tn, tn)][None, :]
        r = jnp.sum(acc * dy_ref[tok, pl.ds(j * tn, tn)].astype(jnp.float32), axis=1)
        plgpu.store(r_ref.at[j, pl.ds(r0, TM_ALL)], r, mask=keep)


def _dot(pln, xs, ws, b, dy, R):
    """Per sorted row of a plan: <dy_t, sum_i x_it W_i[g]^T + b[g]> -> (R,) in sorted order."""
    k, m = sum(x.shape[1] for x in xs), ws[0].shape[1]
    tn, tk, warps, stages = _cfg("dot", k, m, xs[0].dtype)
    tn, tk = min(tn, m), min([tk] + [x.shape[1] for x in xs])
    Rp = pln["tok"].shape[0]
    args = _pairs_args(pln, xs, ws, None) + [dy] + ([b] if b is not None else [])
    r = pl.pallas_call(
        functools.partial(_dot_kernel, n=len(xs), tn=tn, tk=tk, bias=b is not None),
        grid=(m // tn, pln["tile"].shape[0]), in_specs=[_full(a) for a in args],
        out_specs=pl.BlockSpec((m // tn, Rp), lambda j, v: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((m // tn, Rp), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages))(*args)
    return r.sum(0)[:R]


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _routed(xs, ws, b, wts, pln, dtype):
    return _mm(pln["s0"], [x.astype(dtype) for x in xs], [w.astype(dtype) for w in ws], b, xs[0].shape[0])


def _routed_fwd(xs, ws, b, wts, pln, dtype):
    xd, wd = [x.astype(dtype) for x in xs], [w.astype(dtype) for w in ws]
    kinds = ([jnp.zeros((0,), x.dtype) for x in xs], [jnp.zeros((0,), w.dtype) for w in ws])
    return _mm(pln["s0"], xd, wd, b, xs[0].shape[0]), (xd, wd, b, wts, pln, kinds)


def _routed_bwd(dtype, res, dy):
    xd, wd, b, wts, pln, kinds = res
    N, K = wts.shape
    G = wd[0].shape[0]
    dyd, db = _cast_sum(pln["s0"], dy, dtype, G) if b is not None else (dy.astype(dtype), None)
    dxs = tuple(_dx(pln["s0"], dyd, w, N).astype(k.dtype) for w, k in zip(wd, kinds[0]))
    dws = tuple(_dw(pln["s0"], x, dyd, G).astype(k.dtype) for x, k in zip(xd, kinds[1]))
    r = _dot(pln["all"], xd, wd, b, dyd, N * K)
    dwts = jnp.zeros((N * K,), jnp.float32).at[pln["all"]["order"]].set(r).reshape(N, K)
    db = None if b is None else db.astype(b.dtype)
    return dxs, dws, db, dwts.astype(wts.dtype), None


_routed.defvjp(_routed_fwd, _routed_bwd)


def routed_mm(x, w, wts, pln, b=None, dtype=jnp.bfloat16):
    """x (N, k), w (G, m, k), slot weights wts (N, K) (value one-hot on slot 0), b (G, m) or None -> (N, m) fp32.
    y_t = x_t w[l_t0]^T + b[l_t0]; gradients reach x, w, b through slot 0 and wts through every slot.
    x and w may be equal-length tuples: y_t = sum_i x_it w_i[l_t0]^T + b[l_t0] in one kernel, no concat."""
    xs, ws = (tuple(x), tuple(w)) if isinstance(x, (tuple, list)) else ((x,), (w,))
    return _routed(xs, ws, b, wts, pln, dtype)


SQUARINGS = 8  # XLA path: degree-8 Taylor after 2^-SQUARINGS scaling, then squaring
_HI = jax.lax.Precision.HIGHEST
EXPM_WARPS = (2, 1)  # forward, backward
EXPM_THETA = 0.5  # kernels: squarings chosen per block so that |Y|_F 2^-q <= EXPM_THETA
EXPM_QMAX = 16
EXPM_PREC = _HI  # dot precision inside the expm kernels
_C = [1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320]


def _eye(b):
    return (jnp.arange(b)[:, None] == jnp.arange(b)[None, :]).astype(jnp.float32)


def _mmx(prec):
    return lambda a, c: jnp.dot(a, c, precision=prec, preferred_element_type=jnp.float32)


def _scale(y):
    """Squarings q and 2^-q for one block, from its Frobenius norm."""
    nrm = jnp.sqrt(jnp.sum(jnp.sum(y * y, axis=1), axis=0))
    q = jnp.ceil(jnp.log(jnp.maximum(nrm, 1e-30) / EXPM_THETA) * (1 / np.log(2.0)))
    q = jnp.clip(q, 0.0, float(EXPM_QMAX))
    return q.astype(jnp.int32), jnp.exp2(-q)


def _taylor(x, eye, mm):
    """Degree-8 Taylor polynomial in 4 products (Paterson-Stockmeyer): B0 + x^4 (B1 + x^4 / 8!)."""
    x2 = mm(x, x)
    x3 = mm(x2, x)
    x4 = mm(x2, x2)
    b0 = eye + x + _C[2] * x2 + _C[3] * x3
    cc = _C[4] * eye + _C[5] * x + _C[6] * x2 + _C[7] * x3 + _C[8] * x4
    return b0 + mm(x4, cc)


def _taylor_jvp(x, h, eye, mm):
    """The Taylor polynomial at x and its derivative in direction h (12 products)."""
    x2, h2 = mm(x, x), mm(h, x) + mm(x, h)
    x3, h3 = mm(x2, x), mm(h2, x) + mm(x2, h)
    x4, h4 = mm(x2, x2), mm(h2, x2) + mm(x2, h2)
    b0 = eye + x + _C[2] * x2 + _C[3] * x3
    db0 = h + _C[2] * h2 + _C[3] * h3
    cc = _C[4] * eye + _C[5] * x + _C[6] * x2 + _C[7] * x3 + _C[8] * x4
    dcc = _C[5] * h + _C[6] * h2 + _C[7] * h3 + _C[8] * h4
    return b0 + mm(x4, cc), db0 + mm(h4, cc) + mm(x4, dcc)


def _expm_kernel(n_ref, s_ref, d_ref, a_ref, *, prec):
    mm = _mmx(prec)

    @pl.when(pl.program_id(0) < n_ref[0])
    def _():
        s = s_ref[0, :, :]
        y = (s - s.T) * 0.5
        q, sc = _scale(y)
        r = _taylor(y * sc, _eye(s.shape[0]), mm)
        r = jax.lax.fori_loop(0, q, lambda _, r: mm(r, r), r)
        a_ref[0, :, :] = d_ref[0, :][:, None] * r


def _expm_bwd_kernel(n_ref, s_ref, d_ref, g_ref, ds_ref, dd_ref, *, prec):
    """The adjoint of r(Y) = T(Y 2^-q)^(2^q) is its derivative at Y^T = -Y, run forward through the squarings
    (E <- E E, L <- E L + L E) with no stored chain; E ends as r(-Y) = r(Y)^T."""
    mm = _mmx(prec)

    @pl.when(pl.program_id(0) < n_ref[0])
    def _():
        s, d, g = s_ref[0, :, :], d_ref[0, :], g_ref[0, :, :]
        y = (s - s.T) * 0.5
        q, sc = _scale(y)
        e, l = _taylor_jvp(-y * sc, d[:, None] * g * sc, _eye(s.shape[0]), mm)
        e, l = jax.lax.fori_loop(0, q, lambda _, c: (mm(c[0], c[0]), mm(c[0], c[1]) + mm(c[1], c[0])), (e, l))
        dd_ref[0, :] = jnp.sum(g * e.T, axis=1)
        ds_ref[0, :, :] = (l - l.T) * 0.5

    @pl.when(pl.program_id(0) >= n_ref[0])  # rows past the distinct count: zero gradient, written here
    def _():
        dd_ref[0, :] = jnp.zeros(dd_ref.shape[1:], jnp.float32)
        ds_ref[0, :, :] = jnp.zeros(ds_ref.shape[1:], jnp.float32)


def _blk(b, nd):
    return pl.BlockSpec((1, b, b) if nd == 3 else (1, b), (lambda i: (i, 0, 0)) if nd == 3 else (lambda i: (i, 0)))


_N = pl.BlockSpec((1,), lambda i: (0,))


@jax.custom_vjp
def _expm_flat(s, d, n):
    """s (M, b, b), d (M, b); rows from n[0] on are left unwritten."""
    M, b = d.shape
    return pl.pallas_call(functools.partial(_expm_kernel, prec=EXPM_PREC), grid=(M,),
                          in_specs=[_N, _blk(b, 3), _blk(b, 2)], out_specs=_blk(b, 3),
                          out_shape=jax.ShapeDtypeStruct((M, b, b), jnp.float32),
                          compiler_params=plgpu.CompilerParams(num_warps=EXPM_WARPS[0], num_stages=1))(n, s, d)


def _expm_flat_fwd(s, d, n):
    return _expm_flat(s, d, n), (s, d, n)


def _expm_flat_bwd(res, g):
    s, d, n = res
    M, b = d.shape
    ds, dd = pl.pallas_call(
        functools.partial(_expm_bwd_kernel, prec=EXPM_PREC), grid=(M,), in_specs=[_N, _blk(b, 3), _blk(b, 2), _blk(b, 3)],
        out_specs=[_blk(b, 3), _blk(b, 2)],
        out_shape=[jax.ShapeDtypeStruct((M, b, b), jnp.float32), jax.ShapeDtypeStruct((M, b), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=EXPM_WARPS[1], num_stages=1))(n, s, d, g)
    return ds, dd, None


_expm_flat.defvjp(_expm_flat_fwd, _expm_flat_bwd)


def _expm_xla(s, d):
    x = (s - jnp.swapaxes(s, -1, -2)) * (0.5 / 2.0**SQUARINGS)
    mm = lambda a, c: jnp.einsum("...ij,...jk->...ik", a, c, precision=_HI)
    eye = jnp.eye(s.shape[-1], dtype=s.dtype)
    r = eye + x / 8
    for k in range(7, 0, -1):
        r = eye + mm(x, r) / k
    for _ in range(SQUARINGS):
        r = mm(r, r)
    return d[..., None] * r


def rotation_decay(skew, d, n=None):
    """diag(d) expm(S / 2 - S^T / 2) for skew (..., b, b), d (..., b): one fused kernel per block on GPU.
    n: only the first n leading entries are needed (the rest are left undefined on GPU)."""
    if jax.default_backend() != "gpu" or skew.shape[-1] < 16:
        return _expm_xla(skew, d)
    b, lead = skew.shape[-1], skew.shape[:-2]
    per = 1 if n is None else int(np.prod(lead[1:]))
    cnt = jnp.full((1,), int(np.prod(lead)), jnp.int32) if n is None else (n * per).astype(jnp.int32).reshape(1)
    a = _expm_flat(skew.reshape(-1, b, b).astype(jnp.float32), d.reshape(-1, b).astype(jnp.float32), cnt)
    return a.reshape(skew.shape)


def leaf_blocks(skew, d_raw):
    """Each leaf's rotation-decay blocks once per step, (leaves, n / b, b, b), as the scan reads them by leaf id."""
    return rotation_decay(skew, jnp.clip(d_raw, -1.0, 1.0).reshape(skew.shape[:-1]))


def distinct_cap(N, vocab=None):
    """Static row count of the root table for N token positions. On fineweb_edu (GPT-2 ids) a 1024-token window
    batch holds about 5 N^0.7 distinct ids (max over 300 batches 5.45 N^0.7); the cap is 6.5 N^0.7, at most N, vocab."""
    cap = -(-int(6.5 * N**0.7) // 256) * 256
    return int(min(N, cap, vocab or N))


def token_blocks(skew0, d0, g_s, g_d, z, ids, dtype=jnp.bfloat16, vocab=None, max_distinct=None):
    """Root leaf of token x: skew0 + z(x) g_s, d0 + z(x) g_d. Blocks are computed once per distinct token id.
    z (T, B, r), ids (T, B) -> table (U, g, b, b) whose rows past the distinct count are undefined, index (T, B).
    U = max_distinct, else distinct_cap(T B, vocab). Ids past the first U distinct (by value) read row U - 1."""
    N = ids.size
    U = int(min(N, max_distinct)) if max_distinct else distinct_cap(N, vocab)
    flat = ids.reshape(-1)
    order = jnp.argsort(flat).astype(jnp.int32)
    sv = flat[order]
    new = jnp.concatenate([jnp.ones((1,), bool), sv[1:] != sv[:-1]])
    rank = jnp.cumsum(new.astype(jnp.int32)) - 1
    n = jnp.minimum(rank[-1] + 1, U)
    first = order[jnp.nonzero(new, size=U, fill_value=0)[0]]  # one position per distinct id
    inv = jnp.zeros((N,), jnp.int32).at[order].set(jnp.minimum(rank, U - 1))
    zu = z.reshape(N, -1)[first]
    mm = lambda a, c: jnp.dot(a.astype(dtype), c.astype(dtype), preferred_element_type=jnp.float32)
    sk = skew0 + mm(zu, g_s).reshape((U,) + skew0.shape)
    d = jnp.clip(d0 + mm(zu, g_d), -1.0, 1.0).reshape((U,) + skew0.shape[:-1])
    return rotation_decay(sk, d, n), inv.reshape(ids.shape)


@jax.custom_vjp
def tree_sum(levels):
    """Leaf j = (D + 1)^-1/2 sum_d levels[d][j >> (D - d)], levels[d] (2^d, ...). Forward and backward walk the tree
    one level at a time, so the backward reads each leaf gradient once instead of once per level."""
    acc = levels[0]
    for lv in levels[1:]:
        acc = jnp.repeat(acc, 2, axis=0) + lv
    return acc * (len(levels) ** -0.5)


def _tree_sum_fwd(levels):
    return tree_sum(levels), None


def _tree_sum_bwd(_, g):
    D = int(np.log2(g.shape[0]))
    g = g * ((D + 1) ** -0.5)
    out = [g]
    for _ in range(D):
        g = g.reshape((g.shape[0] // 2, 2) + g.shape[1:]).sum(1)
        out.append(g)
    return (out[::-1],)


tree_sum.defvjp(_tree_sum_fwd, _tree_sum_bwd)


def fits(k, m, *more):
    """Shapes the kernels take: every width 16, 32 or a multiple of 64."""
    return all(d in (16, 32) or d % 64 == 0 for d in (k, m) + more)
