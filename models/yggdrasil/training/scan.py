"""Chunked linear scan h_t = A_t h_{t-1} + x_t over 16x16 blocks (Pallas, Triton backend), h_0 = 0.

A_t is read from a table by row id, tab[idx[t, batch], block], so a per-token A is a table of T B rows and a
routed branch reads its leaf table directly. T is cut into chunks of c tokens. With one chunk the kernel is a
plain sequential sweep. With more, a first sweep gives each chunk's product and local end state, a short
associative scan over the T / c chunks gives the start states, and a second sweep writes h. Each program holds
G blocks and their states in registers; no composed product is written per token. fp32 state, any input dtype.

The backward pass is the reverse sweep with transposed blocks, lam_t = g_t + A_{t+1}^T lam_{t+1}, dx_t = lam_t,
dA_t = lam_t h_{t-1}^T. Per-token tables get dA by plain stores in the sweep. Shared tables get dA from a second
kernel: tokens are sorted by row id and each program sums lam h^T over the tokens of its rows, so no atomics;
it also gives the slot-weight gradients of slots past 0 with each row's block held in registers.
Both sweeps load the next step's inputs one step ahead.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

CHUNK = 0  # 0: one sequential sweep over T; else chunk length
GROUP = 2  # blocks per program
WARPS = 1
STAGES = 1
DW_SWEEP = 0  # 1: all slot-weight gradients in the reverse sweep, 0: slot 0 there, the others in _rows
SEG_GROUP = 2  # _rows kernel: blocks per program
SEG_WARPS = 4  # _rows kernel: warps when tiles are longer than 16
HI = jax.lax.Precision.HIGHEST
ROWS_TF32 = 1  # table gradient kernels: small dots in TF32 (about 3e-4 relative) instead of exact fp32
ROWS_TILE = 0  # _rows kernel: rows per tile (0: by entries per row)
SEG_LEN = 512  # routed tables (K > 1): entries per program, so a leaf read by many tokens spans several programs


def _comb(l, r):
    return (jnp.einsum("...ij,...jk->...ik", r[0], l[0], precision=HI),
            jnp.einsum("...ij,...j->...i", r[0], l[1], precision=HI) + r[1])


def _mv(a, h, trans):
    """a (G, b, b), h (G, b) -> a h, or a^T h."""
    return jnp.sum(a * h[:, :, None], axis=1) if trans else jnp.sum(a * h[:, None, :], axis=2)


def _summary_kernel(idx_ref, tab_ref, x_ref, p_ref, s_ref, *, c, G, rev):
    """Chunk map from a zero state. Forward: P = A_e..A_s, s = end state. Reverse: P = A_s^T..A_e^T, s = u_s."""
    k = pl.program_id(2)
    b = x_ref.shape[-1]
    eye = (jnp.arange(b)[:, None] == jnp.arange(b)[None, :]).astype(jnp.float32)

    def body(i, carry):
        ps, s = carry[:-1], carry[-1]
        t = k * c + (c - 1 - i if rev else i)
        r = idx_ref[t, 0]
        a = tab_ref[r].astype(jnp.float32)
        x = x_ref[t, 0].astype(jnp.float32)
        s = _mv(a, x + s, True) if rev else _mv(a, s, False) + x
        out = []
        for g in range(G):  # 16x16 matmuls, one per block
            ag = tab_ref[r, g].astype(jnp.float32)
            if rev:
                out.append(jax.lax.dot_general(ag, ps[g], (((0,), (0,)), ((), ())), precision=HI))
            else:
                out.append(jnp.dot(ag, ps[g], precision=HI))
        return (*out, s)
    carry = jax.lax.fori_loop(0, c, body, (*([eye] * G), jnp.zeros((G, b), jnp.float32)))
    for g in range(G):
        p_ref[0, 0, g] = carry[g]
    s_ref[0, 0] = carry[-1]


def _fwd_kernel(idx_ref, tab_ref, x_ref, hin_ref, h_ref, *, c):
    lo = pl.program_id(2) * c

    def fetch(t):
        return tab_ref[idx_ref[t, 0]].astype(jnp.float32), x_ref[t, 0].astype(jnp.float32)

    def body(i, carry):
        h, a, x = carry
        t = lo + i
        nxt = fetch(jnp.minimum(t + 1, lo + c - 1))  # next step's loads first
        h = _mv(a, h, False) + x
        h_ref[t, 0] = h.astype(h_ref.dtype)
        return (h, *nxt)
    jax.lax.fori_loop(0, c, body, (hin_ref[0, 0], *fetch(lo)))


def _bwd_kernel(slot_ref, tab_ref, g_ref, h_ref, uin_ref, *refs, c, K, mode):
    """Reverse sweep, dx = lam. mode "dw": dw_q = <lam h_{t-1}^T, tab[slot q]> summed over the G blocks;
    "dw0": the same for slot 0 only (its row is loaded anyway; dA and the other slots from _rows); "own": dA
    stored by row."""
    dx_ref = refs[0]
    lo = pl.program_id(2) * c
    f32 = jnp.float32
    nq = K if mode == "dw" else 1

    def fetch(t):
        rows = tuple(tab_ref[slot_ref[t, 0, q]].astype(f32) for q in range(nq))
        hp = h_ref[jnp.maximum(t - 1, 0), 0].astype(f32) * (t > 0).astype(f32)
        return (*rows, g_ref[t, 0].astype(f32), hp)

    def body(i, carry):
        u, rows, gt = carry[0], carry[1:1 + nq], carry[1 + nq]
        t = lo + c - 1 - i
        nxt = fetch(jnp.maximum(t - 1, lo))  # next step's loads first
        lam = gt + u
        dx_ref[t, 0] = lam.astype(dx_ref.dtype)
        da = lam[:, :, None] * carry[2 + nq][:, None, :]
        if mode == "own":
            refs[1][slot_ref[t, 0, 0]] = da.astype(refs[1].dtype)
        else:
            for q in range(nq):
                refs[1][t, 0, 0, q] = jnp.sum(da * rows[q])
        return (_mv(rows[0], lam, True), *nxt)
    jax.lax.fori_loop(0, c, body, (uin_ref[0, 0], *fetch(lo + c - 1)))


def _rows_kernel(nnz_ref, nz_ref, off_ref, order_ref, tab_ref, dx_ref, h_ref, _, out_ref, *dw_ref, R, P, B, K):
    """Entries e = token p, slot q (e = p K + q) sorted by row id; off[l] is row l's first. nz lists the nnz rows
    with entries; program i owns nz[i P .. i P + P - 1] and walks their entries in tiles of R: dtab[l] = sum over
    slot-0 entries of lam_p h_{p-B}^T (out_ref starts as zeros, so unread rows stay zero), and with dw_ref,
    dw[e] = lam_p^T tab[l] h_{p-B} over the G blocks for slots q > 0 (zero for slot 0, which the sweep gives)."""
    G, b = dx_ref.shape[1], dx_ref.shape[2]
    prec = None if ROWS_TF32 else HI
    lo = pl.program_id(0) * P
    ar = jnp.arange(R, dtype=jnp.int32)
    col = jnp.arange(b, dtype=jnp.int32)[None, :]
    f32 = jnp.float32

    def row(j, carry):
        l = nz_ref[j]
        s, e = off_ref[l], off_ref[l + 1]
        a = [tab_ref[l, g].astype(f32) for g in range(G)] if dw_ref else None

        def tile(k, acc):
            pos = s + k * R + ar
            m = pos < e
            ent = plgpu.load(order_ref.at[pos], mask=m, other=0)
            p = ent // K
            m0 = (m & (ent % K == 0))[:, None]
            mh = (m & (p >= B))[:, None]
            ph = jnp.maximum(p - B, 0)[:, None]
            out, dw = [], jnp.zeros((R,), f32)
            for g in range(G):
                lam = plgpu.load(dx_ref.at[p[:, None], g, col], mask=m[:, None], other=0.0).astype(f32)
                hp = plgpu.load(h_ref.at[ph, g, col], mask=mh, other=0.0).astype(f32)
                l0 = lam if K == 1 else jnp.where(m0, lam, 0.0)
                out.append(acc[g] + jax.lax.dot_general(l0, hp, (((0,), (0,)), ((), ())), precision=prec,
                                                        preferred_element_type=f32))
                if dw_ref:
                    v = jax.lax.dot_general(hp, a[g], (((1,), (1,)), ((), ())), precision=prec,
                                            preferred_element_type=f32)  # tab[l] h, (R, b)
                    dw = dw + jnp.sum(lam * v, axis=1)
            if dw_ref:
                dw = jnp.where(m0[:, 0], 0.0, dw)
                plgpu.store(dw_ref[0].at[jnp.where(m, ent, dw_ref[0].shape[0] - 1), 0], dw, mask=m)  # spare last row
            return tuple(out)
        acc = jax.lax.fori_loop(0, (e - s + R - 1) // R, tile, tuple(jnp.zeros((b, b), f32) for _ in range(G)))
        for g in range(G):
            out_ref[l, g] = acc[g]
        return carry
    jax.lax.fori_loop(lo, jnp.minimum(lo + P, nnz_ref[0]), row, 0)


def _seg_kernel(srow_ref, sbeg_ref, send_ref, order_ref, tab_ref, dx_ref, h_ref, out_ref, *dw_ref, R, B, K, S):
    """One segment of at most S entries of one row (entries e = token p, slot q sorted by row id): partial
    dtab = sum over its slot-0 entries of lam_p h_{p-B}^T, and with dw_ref, dw[e] = lam_p^T tab[l] h_{p-B} for
    slots q > 0. Rows read by many tokens span several segments, so no program walks a long row alone."""
    G, b = dx_ref.shape[1], dx_ref.shape[2]
    prec = None if ROWS_TF32 else HI
    i = pl.program_id(0)
    l, s, e = srow_ref[i], sbeg_ref[i], send_ref[i]
    ar = jnp.arange(R, dtype=jnp.int32)
    col = jnp.arange(b, dtype=jnp.int32)[None, :]
    f32 = jnp.float32
    a = [tab_ref[l, g].astype(f32) for g in range(G)] if dw_ref else None

    def tile(k, acc):
        pos = s + k * R + ar
        m = pos < e
        ent = plgpu.load(order_ref.at[pos], mask=m, other=0)
        p = ent // K
        m0 = (m & (ent % K == 0))[:, None]
        mh = (m & (p >= B))[:, None]
        ph = jnp.maximum(p - B, 0)[:, None]
        out, dw = [], jnp.zeros((R,), f32)
        for g in range(G):
            lam = plgpu.load(dx_ref.at[p[:, None], g, col], mask=m[:, None], other=0.0).astype(f32)
            hp = plgpu.load(h_ref.at[ph, g, col], mask=mh, other=0.0).astype(f32)
            l0 = lam if K == 1 else jnp.where(m0, lam, 0.0)
            out.append(acc[g] + jax.lax.dot_general(l0, hp, (((0,), (0,)), ((), ())), precision=prec,
                                                    preferred_element_type=f32))
            if dw_ref:
                v = jax.lax.dot_general(hp, a[g], (((1,), (1,)), ((), ())), precision=prec,
                                        preferred_element_type=f32)
                dw = dw + jnp.sum(lam * v, axis=1)
        if dw_ref:
            dw = jnp.where(m0[:, 0], 0.0, dw)
            plgpu.store(dw_ref[0].at[jnp.where(m, ent, dw_ref[0].shape[0] - 1), 0], dw, mask=m)
        return tuple(out)
    acc = jax.lax.fori_loop(0, (jnp.maximum(e - s, 0) + R - 1) // R, tile,
                            tuple(jnp.zeros((b, b), f32) for _ in range(G)))
    for g in range(G):
        out_ref[0, g] = acc[g]


def _seg_cfg(N, L):
    """Tile length, rows per program, blocks per program and warps of the _rows kernel."""
    per = N / max(L, 1)
    R = ROWS_TILE or (16 if per < 32 else (32 if per < 128 else 64))
    P = max(1, min(16, L // 2048))
    return R, P, SEG_GROUP, SEG_WARPS if R > 16 else 1


def _rows(tab, slots, dx, h, dw, interpret):
    """slots (T, B, K) row ids, dx and h (T, B, g, b) -> dtab (L, g, b, b) fp32 and, if dw, dw (T, B, K) fp32."""
    if SEG_LEN and dw:  # routed tables; the root (one row per distinct token, K = 1) keeps one program per row
        return _rows_seg(tab, slots, dx, h, dw, interpret)
    T, B, g, b = dx.shape
    L, K = tab.shape[0], slots.shape[-1] if dw else 1
    N = T * B
    R, P, G, w = _seg_cfg(N * K, L)
    G = G if g % G == 0 else 1
    ids = slots[..., :K].reshape(-1)
    order = jnp.argsort(ids, stable=True).astype(jnp.int32)
    cnt = jnp.bincount(ids, length=L)
    off = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(cnt).astype(jnp.int32)])
    nz = jnp.nonzero(cnt > 0, size=L, fill_value=0)[0].astype(jnp.int32)
    nnz = jnp.sum(cnt > 0).astype(jnp.int32).reshape(1)
    vec = pl.BlockSpec((N, G, b), lambda i, j: (0, j, 0))
    outs = [pl.BlockSpec((L, G, b, b), lambda i, j: (0, j, 0, 0))]
    shapes = [jax.ShapeDtypeStruct((L, g, b, b), jnp.float32)]
    if dw:
        outs.append(pl.BlockSpec((N * K + 1, 1), lambda i, j: (0, j)))
        shapes.append(jax.ShapeDtypeStruct((N * K + 1, g // G), jnp.float32))
    res = pl.pallas_call(
        functools.partial(_rows_kernel, R=R, P=P, B=B, K=K), grid=(pl.cdiv(L, P), g // G),
        in_specs=[pl.BlockSpec((1,), lambda i, j: (0,)), pl.BlockSpec((L,), lambda i, j: (0,)),
                  pl.BlockSpec((L + 1,), lambda i, j: (0,)), pl.BlockSpec((N * K,), lambda i, j: (0,)),
                  pl.BlockSpec((L, G, b, b), lambda i, j: (0, j, 0, 0)), vec, vec, outs[0]],
        out_specs=outs, out_shape=shapes, input_output_aliases={7: 0},
        compiler_params=plgpu.CompilerParams(num_warps=w, num_stages=1), interpret=interpret,
    )(nnz, nz, off, order, tab, dx.reshape(N, g, b), h.reshape(N, g, b), jnp.zeros((L, g, b, b), jnp.float32))
    return res[0], (res[1][:-1].sum(1).reshape(T, B, K) if dw else None)


def _rows_seg(tab, slots, dx, h, dw, interpret):
    """_rows with every row's entries cut into segments of at most SEG_LEN, one program per segment; the
    partial sums are added into their rows afterwards."""
    T, B, g, b = dx.shape
    L, K = tab.shape[0], slots.shape[-1] if dw else 1
    N, S = T * B, SEG_LEN
    R = ROWS_TILE or 32
    G = SEG_GROUP if g % SEG_GROUP == 0 else 1
    ids = slots[..., :K].reshape(-1)
    order = jnp.argsort(ids, stable=True).astype(jnp.int32)
    cnt = jnp.bincount(ids, length=L).astype(jnp.int32)
    off = jnp.cumsum(cnt) - cnt
    nseg = (cnt + S - 1) // S
    send_ = jnp.cumsum(nseg)
    n_seg = N * K // S + min(L, N * K)  # bound on the number of segments
    sid = jnp.arange(n_seg, dtype=jnp.int32)
    srow = jnp.minimum(jnp.searchsorted(send_, sid, side="right"), L - 1).astype(jnp.int32)
    k = sid - (send_ - nseg)[srow]
    valid = sid < send_[-1]
    sbeg = jnp.where(valid, off[srow] + k * S, 0).astype(jnp.int32)
    send = jnp.where(valid, jnp.minimum(sbeg + S, off[srow] + cnt[srow]), 0).astype(jnp.int32)
    one = pl.BlockSpec((n_seg,), lambda i, j: (0,))
    vec = pl.BlockSpec((N, G, b), lambda i, j: (0, j, 0))
    outs = [pl.BlockSpec((1, G, b, b), lambda i, j: (i, j, 0, 0))]
    shapes = [jax.ShapeDtypeStruct((n_seg, g, b, b), jnp.float32)]
    if dw:
        outs.append(pl.BlockSpec((N * K + 1, 1), lambda i, j: (0, j)))
        shapes.append(jax.ShapeDtypeStruct((N * K + 1, g // G), jnp.float32))
    res = pl.pallas_call(
        functools.partial(_seg_kernel, R=R, B=B, K=K, S=S), grid=(n_seg, g // G),
        in_specs=[one, one, one, pl.BlockSpec((N * K,), lambda i, j: (0,)),
                  pl.BlockSpec((L, G, b, b), lambda i, j: (0, j, 0, 0)), vec, vec],
        out_specs=outs, out_shape=shapes,
        compiler_params=plgpu.CompilerParams(num_warps=SEG_WARPS if R > 16 else 1, num_stages=1), interpret=interpret,
    )(srow, sbeg, send, order, tab, dx.reshape(N, g, b), h.reshape(N, g, b))
    dtab = jnp.zeros((L, g, b, b), jnp.float32).at[srow].add(jnp.where(valid[:, None, None, None], res[0], 0.0))
    return dtab, (res[1][:-1].sum(1).reshape(T, B, K) if dw else None)


def _call(kernel, grid, in_specs, out_specs, out_shape, warps, **kw):
    w, st = warps if isinstance(warps, tuple) else (warps, STAGES)
    return pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
                          compiler_params=plgpu.CompilerParams(num_warps=w, num_stages=st), **kw)


def _specs(T, L, G, b):
    idx = pl.BlockSpec((T, 1), lambda i, j, k: (0, i))
    tab = pl.BlockSpec((L, G, b, b), lambda i, j, k: (0, j, 0, 0))
    vec = pl.BlockSpec((T, 1, G, b), lambda i, j, k: (0, i, j, 0))
    cvec = pl.BlockSpec((1, 1, G, b), lambda i, j, k: (k, i, j, 0))
    cmat = pl.BlockSpec((1, 1, G, b, b), lambda i, j, k: (k, i, j, 0, 0))
    return idx, tab, vec, cvec, cmat


def _summary(tab, idx, x, rev, c, G, warps, interpret):
    T, B, g, b = x.shape
    nc = T // c
    si, st, sv, scv, scm = _specs(T, tab.shape[0], G, b)
    return _call(functools.partial(_summary_kernel, c=c, G=G, rev=rev), (B, g // G, nc), [si, st, sv], [scm, scv],
                 [jax.ShapeDtypeStruct((nc, B, g, b, b), jnp.float32),
                  jax.ShapeDtypeStruct((nc, B, g, b), jnp.float32)], warps, interpret=interpret)(idx, tab, x)


def _forward(tab, idx, x, c, G, warps, interpret):
    T, B, g, b = x.shape
    nc = T // c
    if nc == 1:
        hin = jnp.zeros((1, B, g, b), jnp.float32)
    else:
        p, s = _summary(tab, idx, x, False, c, G, warps, interpret)
        end = jax.lax.associative_scan(_comb, (p, s), axis=0)[1]
        hin = jnp.concatenate([jnp.zeros_like(end[:1]), end[:-1]])
    si, st, sv, scv, _ = _specs(T, tab.shape[0], G, b)
    return _call(functools.partial(_fwd_kernel, c=c), (B, g // G, nc), [si, st, sv, scv], sv,
                 jax.ShapeDtypeStruct(x.shape, x.dtype), warps, interpret=interpret)(idx, tab, x, hin)


def _backward(tab, slots, h, gh, own, c, G, warps, interpret):
    T, B, g, b = h.shape
    K, nc = slots.shape[-1], T // c
    idx = slots[..., 0]
    if nc == 1:
        uin = jnp.zeros((1, B, g, b), jnp.float32)
    else:
        p, s = _summary(tab, idx, gh, True, c, G, warps, interpret)
        mu = jax.lax.associative_scan(_comb, (p[::-1], s[::-1]), axis=0)[1][::-1]  # u at each chunk start
        uin = jnp.concatenate([mu[1:], jnp.zeros_like(mu[:1])])
    _, st, sv, scv, _ = _specs(T, tab.shape[0], G, b)
    ss = pl.BlockSpec((T, 1, K), lambda i, j, k: (0, i, 0))
    mode = "own" if own else ("dw" if DW_SWEEP or K == 1 else "dw0")
    nq = K if mode == "dw" else 1
    outs, shapes = [sv], [jax.ShapeDtypeStruct(gh.shape, gh.dtype)]
    if mode != "own":
        outs.append(pl.BlockSpec((T, 1, 1, nq), lambda i, j, k: (0, i, j, 0)))
        shapes.append(jax.ShapeDtypeStruct((T, B, g // G, nq), jnp.float32))
    if mode == "own":
        outs.append(st)
        shapes.append(jax.ShapeDtypeStruct(tab.shape, tab.dtype))
    res = _call(functools.partial(_bwd_kernel, c=c, K=K, mode=mode), (B, g // G, nc), [ss, st, sv, sv, scv], outs,
                shapes, warps, interpret=interpret)(slots, tab, gh, h, uin)
    if mode == "own":  # per-token rows: w is a constant one
        return res[0], jnp.zeros(slots.shape, jnp.float32), res[1]
    if mode == "dw":
        return res[0], res[1].sum(2), _rows(tab, slots, res[0], h, False, interpret)[0]
    dtab, dw = _rows(tab, slots, res[0], h, True, interpret)
    return res[0], dw.at[..., 0].set(res[1].sum((2, 3))), dtab


def _cfg(T, g, chunk, group, back=False):
    chunk = CHUNK if chunk is None else chunk
    if isinstance(chunk, tuple):  # (forward, backward)
        chunk = chunk[back]
    group = GROUP if group is None else group
    c = chunk if chunk and T % chunk == 0 else T
    G = group if g % group == 0 else 1
    return c, G


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _core(tab, slots, w, x, own, chunk, group, warps, interpret):
    c, G = _cfg(x.shape[0], x.shape[2], chunk, group)
    return _forward(tab, slots[..., 0], x, c, G, warps, interpret)


def _core_fwd(tab, slots, w, x, own, chunk, group, warps, interpret):
    h = _core(tab, slots, w, x, own, chunk, group, warps, interpret)
    return h, (tab, slots, w, h)


def _core_bwd(own, chunk, group, warps, interpret, res, gh):
    tab, slots, w, h = res
    c, G = _cfg(h.shape[0], h.shape[2], chunk, group, True)
    dx, dw, dtab = _backward(tab, slots, h, gh.astype(h.dtype), own, c, G, warps, interpret)
    return dtab.astype(tab.dtype), None, dw.astype(w.dtype), dx


_core.defvjp(_core_fwd, _core_bwd)


def table_scan(tab, slots, w, x, chunk=None, group=None, warps=WARPS, interpret=False):
    """tab (L, g, b, b), slots (T, B, K) row ids with slot 0 the one used, w (T, B, K) straight-through weights
    (value one-hot on slot 0, gradient <dA_t, tab[slot]>), x (T, B, g, b) -> h (T, B, g, b) in x.dtype.
    Default: two blocks per program; the backward sweep in chunks of 64 when B <= 8."""
    if chunk is None:
        chunk = (0, 64 if x.shape[1] <= 8 else 0)
    group = 2 if group is None else group
    return _core(tab, slots.astype(jnp.int32), w, x, False, chunk, group, warps, interpret)


def scan(a, x, chunk=None, group=None, warps=WARPS, interpret=False):
    """Per-token transitions: a (T, ..., g, b, b), x (T, ..., g, b) -> h in x.dtype. Drop-in for an associative scan."""
    T, g, b = x.shape[0], x.shape[-2], x.shape[-1]
    shape = x.shape
    x = x.reshape(T, -1, g, b)
    B = x.shape[1]
    ids = (jnp.arange(T)[:, None] * B + jnp.arange(B)[None, :]).astype(jnp.int32)[..., None]
    h = _core(a.reshape(T * B, g, b, b), ids, jnp.ones(ids.shape, jnp.float32), x, True, chunk, group, warps,
              interpret)
    return h.reshape(shape)

