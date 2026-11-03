"""Grouped matmul without padding (Pallas, Triton backend), as in MegaBlocks / megablox: y_i = x_i W[g_i].

Rows are sorted by group. The sorted rows are cut into tiles of TM rows; a tile that holds rows of several groups is
visited once per group, with the rows of other groups masked, so no row is padded or dropped and every visit reads
its weight tile straight from W[g]. The weight gradient visits, for each group, the tiles that hold its rows.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

TM, TK, TN = 64, 64, 64


def _visits(start, end, n_tiles):
    """(tile, group) pairs: every tile that holds rows of a group, padded to n_tiles + G with empty visits."""
    G = start.shape[0]
    first = start // TM
    last = jnp.where(end > start, (end - 1) // TM, first - 1)
    n = jnp.maximum(last - first + 1, 0)
    off = jnp.cumsum(n) - n
    size = n_tiles + G
    v = jnp.arange(size)
    grp = jnp.clip(jnp.searchsorted(off + n, v, side="right"), 0, G - 1)
    tile = jnp.clip(first[grp] + v - off[grp], 0, n_tiles - 1)
    valid = v < (off + n)[-1]
    return tile.astype(jnp.int32), jnp.where(valid, grp, -1).astype(jnp.int32)


def _gmm_kernel(tile_ref, grp_ref, start_ref, end_ref, x_ref, w_ref, y_ref, *, k, trans):
    v, j = pl.program_id(0), pl.program_id(1)
    gr = grp_ref[v]
    g = jnp.maximum(gr, 0)
    rows = tile_ref[v] * TM + jnp.arange(TM)
    keep = (rows >= start_ref[g]) & (rows < end_ref[g]) & (gr >= 0)
    acc = jnp.zeros((TM, TN), jnp.float32)
    for kk in range(k // TK):
        xb = x_ref[pl.ds(tile_ref[v] * TM, TM), pl.ds(kk * TK, TK)]
        if trans:
            wb = w_ref[g, pl.ds(j * TN, TN), pl.ds(kk * TK, TK)].T
        else:
            wb = w_ref[g, pl.ds(kk * TK, TK), pl.ds(j * TN, TN)]
        acc += jnp.dot(xb, wb, preferred_element_type=jnp.float32)
    plgpu.store(y_ref.at[pl.ds(tile_ref[v] * TM, TM), pl.ds(j * TN, TN)], acc.astype(y_ref.dtype), mask=keep[:, None])


def _tgmm_kernel(start_ref, end_ref, x_ref, dy_ref, dw_ref, *, n_tiles):
    g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    s, e = start_ref[g], end_ref[g]

    def body(t, acc):
        rows = t * TM + jnp.arange(TM)
        keep = ((rows >= s) & (rows < e))[:, None]
        xb = jnp.where(keep, x_ref[pl.ds(t * TM, TM), pl.ds(i * TK, TK)], 0)
        db = jnp.where(keep, dy_ref[pl.ds(t * TM, TM), pl.ds(j * TN, TN)], 0)
        return acc + jnp.dot(xb.T, db, preferred_element_type=jnp.float32)
    t0, t1 = s // TM, jnp.where(e > s, (e - 1) // TM + 1, s // TM)
    acc = jax.lax.fori_loop(t0, jnp.minimum(t1, n_tiles), body, jnp.zeros((TK, TN), jnp.float32))
    dw_ref[0, :, :] = acc.astype(dw_ref.dtype)


def _gmm(x, w, start, end, trans, interpret):
    R, k = x.shape
    m = w.shape[1] if trans else w.shape[2]
    n_tiles = R // TM
    tile, grp = _visits(start, end, n_tiles)
    full = lambda a: pl.BlockSpec(a.shape, lambda v, j: (0,) * a.ndim)
    args = (tile, grp, start, end, x, w)
    return pl.pallas_call(functools.partial(_gmm_kernel, k=k, trans=trans), grid=(tile.shape[0], m // TN),
                          in_specs=[full(a) for a in args], out_specs=pl.BlockSpec((R, m), lambda v, j: (0, 0)),
                          out_shape=jax.ShapeDtypeStruct((R, m), jnp.float32), interpret=interpret)(*args)


def _tgmm(x, dy, start, end, interpret):
    R, k = x.shape
    m, G = dy.shape[1], start.shape[0]
    full = lambda a: pl.BlockSpec(a.shape, lambda g, i, j: (0,) * a.ndim)
    args = (start, end, x, dy)
    return pl.pallas_call(functools.partial(_tgmm_kernel, n_tiles=R // TM), grid=(G, k // TK, m // TN),
                          in_specs=[full(a) for a in args], out_specs=pl.BlockSpec((1, TK, TN), lambda g, i, j: (g, i, j)),
                          out_shape=jax.ShapeDtypeStruct((G, k, m), jnp.float32), interpret=interpret)(*args)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def gmm_sorted(x, w, start, end, interpret=False):
    """x (R, k) rows sorted by group, w (G, k, m), group g owns rows [start_g, end_g) -> (R, m), fp32 accumulation."""
    return _gmm(x, w, start, end, False, interpret)


def _fwd(x, w, start, end, interpret):
    return _gmm(x, w, start, end, False, interpret), (x, w, start, end)


def _bwd(interpret, res, dy):
    x, w, start, end = res
    dy = dy.astype(x.dtype)
    dx = _gmm(dy, w, start, end, True, interpret)
    dw = _tgmm(x, dy, start, end, interpret).astype(w.dtype)
    return dx, dw, None, None


gmm_sorted.defvjp(_fwd, _bwd)


def fits(R, k, m):
    return R % TM == 0 and k % TK == 0 and m % TN == 0


def grouped_mm(x, gid, w, dtype=jnp.bfloat16, interpret=False):
    """x (R, k), group ids gid (R,), w (G, k, m) -> (R, m) in fp32: sort rows by group, gmm_sorted, unsort.
    Rows with gid >= G belong to no group and give zeros."""
    G = w.shape[0]
    gid = jnp.minimum(gid, G)
    order = jnp.argsort(gid)
    count = jnp.bincount(gid, length=G + 1)[:G]
    end = jnp.cumsum(count).astype(jnp.int32)
    y = gmm_sorted(x[order].astype(dtype), w.astype(dtype), end - count.astype(jnp.int32), end, interpret)
    y = jnp.where((jnp.arange(y.shape[0]) < end[-1])[:, None], y, 0.0)
    return jnp.zeros(y.shape, jnp.float32).at[order].set(y)
