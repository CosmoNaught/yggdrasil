"""The block-diagonal linear recurrence: an associative scan, and on a GPU a sweep that reads A from the expert table.

expert_scan (Pallas, Triton backend) runs h_t = A_{r_t} h_{t-1} + x_t with A_r read from the table by expert id, so
no per-token A is formed. One program holds SCAN_GROUP blocks of one sequence in registers and sweeps T; the next
step's loads are issued before the current step's update. The backward pass is the reverse sweep with transposed
blocks, lam_t = g_t + A_{t+1}^T lam_{t+1}, dx_t = lam_t, which also gives the slot-0 weight gradient
<lam_t h_{t-1}^T, A_{r_t}> from the block it holds. The table gradient sum_t lam_t h_{t-1}^T and the gradients of
the other slots come from a second kernel over the (token, slot) entries sorted by expert, cut into segments of at
most SEG_LEN entries so that an expert read by many tokens does not run on one program; no atomics.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

HI = jax.lax.Precision.HIGHEST
SCAN_GROUP = 2  # blocks per program in the sweeps
SCAN_WARPS = 1
SEG_GROUP = 2  # blocks per program in the table-gradient kernel
SEG_TILE = 32  # entries per tile in the table-gradient kernel
SEG_WARPS = 4
SEG_LEN = 512  # entries per table-gradient program
MIN_BATCH = 8  # smaller batches leave the sweep with too few programs; they take the associative scan


def _combine(l, r):
    return (jnp.einsum("...ij,...jk->...ik", r[0], l[0]),
            jnp.einsum("...ij,...j->...i", r[0], l[1]) + r[1])


def scan(a, x, h0=None):
    """h_t = a_t h_{t-1} + x_t: a (T, B, g, b, b), x (T, B, g, b), h0 (B, g * b) or None (zeros) -> h (T, B, g, b)."""
    if h0 is not None:
        x = x.at[0].add(jnp.einsum("b...ij,b...j->b...i", a[0], h0.reshape(x.shape[1:])))
    return jax.lax.associative_scan(_combine, (a, x), axis=0)[1]


def _mv(a, h, trans):
    """a (G, b, b), h (G, b) -> a h, or a^T h."""
    return jnp.sum(a * h[:, :, None], axis=1) if trans else jnp.sum(a * h[:, None, :], axis=2)


def _fwd_kernel(expert_ref, table_ref, x_ref, h_ref):
    """h (T, 1, G, b) of one sequence and G blocks: h_t = A_{r_t} h_{t-1} + x_t, h_0 = 0."""
    T = x_ref.shape[0]

    def fetch(t):
        return table_ref[expert_ref[t, 0]], x_ref[t, 0]

    def body(t, carry):
        h, a, x = carry
        nxt = fetch(jnp.minimum(t + 1, T - 1))
        h = _mv(a, h, False) + x
        h_ref[t, 0] = h
        return (h, *nxt)
    jax.lax.fori_loop(0, T, body, (jnp.zeros(x_ref.shape[2:], jnp.float32), *fetch(0)))


def _bwd_kernel(expert_ref, table_ref, g_ref, h_ref, dx_ref, dw0_ref):
    """Reverse sweep of one sequence and G blocks: dx (T, 1, G, b) = lam, dw0 (T, 1, 1, 1) = the blocks' share of
    <lam_t h_{t-1}^T, A_{r_t}>."""
    T = g_ref.shape[0]

    def fetch(t):
        hp = h_ref[jnp.maximum(t - 1, 0), 0] * jnp.asarray(t > 0, jnp.float32)
        return table_ref[expert_ref[t, 0, 0]], g_ref[t, 0], hp

    def body(i, carry):
        u, a, g, hp = carry
        t = T - 1 - i
        nxt = fetch(jnp.maximum(t - 1, 0))
        lam = g + u
        dx_ref[t, 0] = lam
        dw0_ref[t, 0, 0, 0] = jnp.sum(lam[:, :, None] * hp[:, None, :] * a)
        return (_mv(a, lam, True), *nxt)
    jax.lax.fori_loop(0, T, body, (jnp.zeros(g_ref.shape[2:], jnp.float32), *fetch(T - 1)))


def _seg_kernel(seg_expert_ref, seg_begin_ref, seg_end_ref, order_ref, table_ref, dx_ref, h_ref, out_ref, dw_ref, *,
                B, k):
    """One segment of the (token p, slot q) entries e = p k + q sorted by expert: out (1, G, b, b) = sum over its
    slot-0 entries of lam_p h_{p-B}^T; dw[e] = <lam_p h_{p-B}^T, A> over the program's G blocks for slots q > 0."""
    G, b = dx_ref.shape[1], dx_ref.shape[2]
    i = pl.program_id(0)
    r, s, e = seg_expert_ref[i], seg_begin_ref[i], seg_end_ref[i]
    ar = jnp.arange(SEG_TILE, dtype=jnp.int32)
    col = jnp.arange(b, dtype=jnp.int32)[None, :]
    a = [table_ref[r, g] for g in range(G)]

    def tile(t, acc):
        pos = s + t * SEG_TILE + ar
        m = pos < e
        ent = plgpu.load(order_ref.at[pos], mask=m, other=0)
        p = ent // k
        m0 = (m & (ent % k == 0))[:, None]
        mh = (m & (p >= B))[:, None]
        ph = jnp.maximum(p - B, 0)[:, None]
        out, dw = [], jnp.zeros((SEG_TILE,), jnp.float32)
        for g in range(G):
            lam = plgpu.load(dx_ref.at[p[:, None], g, col], mask=m[:, None], other=0.0)
            hp = plgpu.load(h_ref.at[ph, g, col], mask=mh, other=0.0)
            out.append(acc[g] + jax.lax.dot_general(jnp.where(m0, lam, 0.0), hp, (((0,), (0,)), ((), ())),
                                                    precision=HI, preferred_element_type=jnp.float32))
            v = jax.lax.dot_general(hp, a[g], (((1,), (1,)), ((), ())), precision=HI,
                                    preferred_element_type=jnp.float32)  # A h, (SEG_TILE, b)
            dw = dw + jnp.sum(lam * v, axis=1)
        dw = jnp.where(m0[:, 0], 0.0, dw)
        plgpu.store(dw_ref.at[jnp.where(m, ent, dw_ref.shape[0] - 1), 0], dw, mask=m)  # spare last row
        return tuple(out)
    acc = jax.lax.fori_loop(0, (jnp.maximum(e - s, 0) + SEG_TILE - 1) // SEG_TILE, tile,
                            tuple(jnp.zeros((b, b), jnp.float32) for _ in range(G)))
    for g in range(G):
        out_ref[0, g] = acc[g]


def _table_grads(table, expert, dx, h):
    """expert (T, B, k), dx and h (T, B, g, b) -> d table (E, g, b, b), slot weight gradients (T, B, k), slot 0 zero."""
    T, B, g, b = dx.shape
    E, k = table.shape[0], expert.shape[-1]
    N, G = T * B, SEG_GROUP if g % SEG_GROUP == 0 else 1
    ids = expert.reshape(-1)
    order = jnp.argsort(ids, stable=True).astype(jnp.int32)
    count = jnp.bincount(ids, length=E).astype(jnp.int32)
    first = jnp.cumsum(count) - count
    n_seg_of = (count + SEG_LEN - 1) // SEG_LEN
    seg_end_of = jnp.cumsum(n_seg_of)
    n_seg = N * k // SEG_LEN + min(E, N * k)  # bound on the number of segments
    sid = jnp.arange(n_seg, dtype=jnp.int32)
    seg_expert = jnp.minimum(jnp.searchsorted(seg_end_of, sid, side="right"), E - 1).astype(jnp.int32)
    valid = sid < seg_end_of[-1]
    begin = jnp.where(valid, first[seg_expert] + (sid - (seg_end_of - n_seg_of)[seg_expert]) * SEG_LEN, 0)
    end = jnp.where(valid, jnp.minimum(begin + SEG_LEN, first[seg_expert] + count[seg_expert]), 0)
    one = pl.BlockSpec((n_seg,), lambda i, j: (0,))
    vec = pl.BlockSpec((N, G, b), lambda i, j: (0, j, 0))
    part, dw = pl.pallas_call(
        functools.partial(_seg_kernel, B=B, k=k), grid=(n_seg, g // G),
        in_specs=[one, one, one, pl.BlockSpec((N * k,), lambda i, j: (0,)),
                  pl.BlockSpec((E, G, b, b), lambda i, j: (0, j, 0, 0)), vec, vec],
        out_specs=[pl.BlockSpec((1, G, b, b), lambda i, j: (i, j, 0, 0)),
                   pl.BlockSpec((N * k + 1, 1), lambda i, j: (0, j))],
        out_shape=[jax.ShapeDtypeStruct((n_seg, g, b, b), jnp.float32),
                   jax.ShapeDtypeStruct((N * k + 1, g // G), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=SEG_WARPS, num_stages=1),
    )(seg_expert, begin.astype(jnp.int32), end.astype(jnp.int32), order, table, dx.reshape(N, g, b),
      h.reshape(N, g, b))
    dtable = jnp.zeros(table.shape, jnp.float32).at[seg_expert].add(jnp.where(valid[:, None, None, None], part, 0.0))
    return dtable, dw[:-1].sum(1).reshape(T, B, k)


def _sweep(kernel, expert, table, *xs, outs):
    """One program per (sequence, SCAN_GROUP blocks); outs: "vec" (T, B, g, b) or "dw0" (T, B, g / G, 1)."""
    T, B, g, b = xs[0].shape
    G = SCAN_GROUP if g % SCAN_GROUP == 0 else 1
    vec = pl.BlockSpec((T, 1, G, b), lambda i, j: (0, i, j, 0))
    specs = {"vec": vec, "dw0": pl.BlockSpec((T, 1, 1, 1), lambda i, j: (0, i, j, 0))}
    shapes = {"vec": jax.ShapeDtypeStruct((T, B, g, b), jnp.float32),
              "dw0": jax.ShapeDtypeStruct((T, B, g // G, 1), jnp.float32)}
    ex = pl.BlockSpec((T, 1) + expert.shape[2:], lambda i, j: (0, i) + (0,) * (expert.ndim - 2))
    return pl.pallas_call(
        kernel, grid=(B, g // G),
        in_specs=[ex, pl.BlockSpec((table.shape[0], G, b, b), lambda i, j: (0, j, 0, 0))] + [vec] * len(xs),
        out_specs=[specs[o] for o in outs], out_shape=[shapes[o] for o in outs],
        compiler_params=plgpu.CompilerParams(num_warps=SCAN_WARPS, num_stages=1))(expert, table, *xs)


@jax.custom_vjp
def _expert_scan(table, expert, w, x):
    return _sweep(_fwd_kernel, expert[..., 0], table, x, outs=["vec"])[0]


def _expert_scan_fwd(table, expert, w, x):
    h = _expert_scan(table, expert, w, x)
    return h, (table, expert, w, h)


def _expert_scan_bwd(res, gh):
    table, expert, w, h = res
    dx, dw0 = _sweep(_bwd_kernel, expert, table, gh.astype(jnp.float32), h, outs=["vec", "dw0"])
    dtable, dw = _table_grads(table, expert, dx, h)
    return dtable, None, dw.at[..., 0].set(dw0.sum((2, 3))).astype(w.dtype), dx


_expert_scan.defvjp(_expert_scan_fwd, _expert_scan_bwd)


def expert_scan(table, expert, w, x, h0=None):
    """h_t = A_t h_{t-1} + x_t with A_t = table[expert[t, :, 0]]: table (E, g, b, b), expert and the straight-through
    weights w (T, B, k) (value one-hot on slot 0; gradient of slot q: <dA_t, table[expert slot q]>), x (T, B, g, b),
    h0 (B, g * b) or None -> h (T, B, g, b): the values and gradients of scan(einsum(w, table[expert]), x, h0)."""
    if h0 is not None:
        x = x.at[0].add(jnp.einsum("b...ij,b...j->b...i", table[expert[0, :, 0]], h0.reshape(x.shape[1:])))
    if jax.default_backend() != "gpu" or x.shape[1] < MIN_BATCH:
        return scan(jnp.einsum("tbq,tbq...->tb...", w, table[expert]), x)
    return _expert_scan(table.astype(jnp.float32), expert.astype(jnp.int32), w, x.astype(jnp.float32))
