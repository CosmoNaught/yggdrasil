"""Expert matmuls of the routed layers without padding (Pallas, Triton backend), in the manner of MegaBlocks.

Each token t reads K experts r_tq with straight-through weights w_tq whose value is one-hot on slot 0, so

    y_t = x_t W[r_t0]^T + b[r_t0]      (value: slot 0 only)
    dw_tq = <dy_t, x_t W[r_tq]^T + b[r_tq]>,  dx_t = dy_t W[r_t0],  dW[g] = sum_{t: r_t0 = g} dy_t^T x_t,

the values and gradients of the per-token gathers sum_q w_tq (x_t W[r_tq]^T + b[r_tq]) in model.py. Rows are sorted
by expert once per layer (plan). The sorted rows are cut into tiles of TM rows, and a tile that holds rows of several
experts is visited once per expert with the other rows masked, so no row is padded or dropped. The kernels gather
token rows by index and scatter results back in token order. The slot weights' gradient reduces each visit's tile
against dy to one number per row. dW splits an expert's rows into chunks of CH rows, so an expert that holds most
tokens does not run on a few programs alone. Several (x, W) pairs with one output run as one kernel, without a
concat. Operands and accumulation are fp32 (IEEE dots).
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

HI = jax.lax.Precision.HIGHEST
TM, TM_ALL = 64, 128  # rows per tile: slot 0 plan, all-slot plan (slot weight gradient only)
CH = 2048  # dW: rows per chunk; an expert with more rows is split into chunks whose partial sums are added after
TILES = {"mm": (64, 32, 4, 2), "dx": (64, 32, 4, 2), "dot": (64, 32, 4, 2),  # TN, TK, warps, stages
         "dw": (64, 64, 32, 4, 2)}  # TN, TK, rows per step, warps, stages


def fits(*widths):
    """Widths the kernels take: 16, 32 or a multiple of 64."""
    return all(d in (16, 32) or d % 64 == 0 for d in widths)


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


def plan(expert, G):
    """expert (N, K) -> sort plans for slot 0 (values, dx, dW) and for all slots (slot weight gradient)."""
    N, K = expert.shape
    expert = jax.lax.stop_gradient(expert).astype(jnp.int32)
    s0 = _segments(expert[:, 0], G)
    s0["cgrp"], s0["chunk"], s0["owner"] = _chunks(s0["start"], s0["end"], N)
    return dict(s0=s0, all=_segments(expert.reshape(-1), G, K, TM_ALL))


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
                                 precision=HI, preferred_element_type=jnp.float32)
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
    tn, tk, warps, stages = TILES["mm"]
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
                                 precision=HI, preferred_element_type=jnp.float32)
        acc = jax.lax.fori_loop(0, m // tk, body, jnp.zeros((TM, tn), jnp.float32))
        plgpu.store(dx_ref.at[tok, pl.ds(j * tn, tn)], acc.astype(dx_ref.dtype), mask=keep[:, None])


def _dx(pln, dy, w, N):
    """dy (N, m), w (G, m, k) -> dx (N, k) fp32, dx_t = dy_t W[g_t]."""
    m, k = w.shape[1], w.shape[2]
    tn, tk, warps, stages = TILES["dx"]
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
            return acc + jnp.dot(dy_ref[tok, pl.ds(i * tn, tn)].T, xb, precision=HI,
                                       preferred_element_type=jnp.float32)
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
    tn, tk, rs, warps, stages = TILES["dw"]
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


def _bias_grad(pln, dy, G):
    """db (G, m) = sum of dy over the rows whose slot-0 expert is g."""
    return jax.ops.segment_sum(dy, pln["ids"], G)


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
    tn, tk, warps, stages = TILES["dot"]
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


@jax.custom_vjp
def _routed(xs, ws, b, wts, pln):
    return _mm(pln["s0"], xs, ws, b, xs[0].shape[0])


def _routed_fwd(xs, ws, b, wts, pln):
    return _routed(xs, ws, b, wts, pln), (xs, ws, b, wts, pln)


def _routed_bwd(res, dy):
    xs, ws, b, wts, pln = res
    N, K = wts.shape
    G = ws[0].shape[0]
    db = _bias_grad(pln["s0"], dy, G) if b is not None else None
    dxs = tuple(_dx(pln["s0"], dy, w, N) for w in ws)
    dws = tuple(_dw(pln["s0"], x, dy, G) for x in xs)
    r = _dot(pln["all"], xs, ws, b, dy, N * K)
    dwts = jnp.zeros((N * K,), jnp.float32).at[pln["all"]["order"]].set(r).reshape(N, K)
    return dxs, dws, db, dwts.astype(wts.dtype), None


_routed.defvjp(_routed_fwd, _routed_bwd)


def routed_mm(x, w, wts, pln, b=None):
    """x (N, k), w (E, m, k), slot weights wts (N, K) (value one-hot on slot 0), b (E, m) or None -> (N, m), fp32.
    y_t = x_t w[r_t0]^T + b[r_t0]; gradients reach x, w, b through slot 0 and wts through every slot.
    x and w may be equal-length tuples: y_t = sum_i x_it w_i[r_t0]^T + b[r_t0] in one kernel, no concat."""
    xs, ws = (tuple(x), tuple(w)) if isinstance(x, (tuple, list)) else ((x,), (w,))
    f32 = lambda t: tuple(a.astype(jnp.float32) for a in t)
    return _routed(f32(xs), f32(ws), b, wts, pln)


