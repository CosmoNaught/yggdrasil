"""Blocked parallel scan h_t = A_t h_{t-1} + x_t with A_t read from a leaf table by id (Pallas, Triton backend).

The window is cut into chunks. Each chunk's (product, end state) comes from one sweep, an associative scan over the
chunks gives every chunk's start state, and a second sweep writes the states. Every program keeps one 16x16 block
and its state in registers, and reads A_t from the table (L2 resident for a tree of leaves), so no per-token A is
stored. The backward pass runs the adjoint recurrence the same way and returns the table gradient (atomic adds) and,
for the straight-through slots, the gradient of each slot weight, <dA_t, A_slot>.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

CHUNK = 64


def _comb(l, r):
    return (jnp.einsum("...ij,...jk->...ik", r[0], l[0]), jnp.einsum("...ij,...j->...i", r[0], l[1]) + r[1])


def _matvec(a, h, trans):
    return jnp.sum(a * h[:, None], axis=0) if trans else jnp.sum(a * h[None, :], axis=1)


def _summary_kernel(idx_ref, tab_ref, x_ref, p_ref, s_ref, *, c, rev):
    """Chunk summary from a zero state: forward P = A_e..A_s, s = local end state; rev: P = A_s^T..A_e^T, s = u_s."""
    b = x_ref.shape[-1]
    eye = (jnp.arange(b)[:, None] == jnp.arange(b)[None, :]).astype(jnp.float32)

    def body(i, carry):
        p, s = carry
        t = c - 1 - i if rev else i
        a = tab_ref[idx_ref[t, 0], 0, :, :]
        x = x_ref[t, 0, 0, :]
        if rev:
            s = _matvec(a, x + s, True)
            p = jnp.sum(a[:, :, None] * p[:, None, :], axis=0)  # A^T P
        else:
            s = _matvec(a, s, False) + x
            p = jnp.sum(a[:, :, None] * p[None, :, :], axis=1)  # A P
        return p, s
    p, s = jax.lax.fori_loop(0, c, body, (eye, jnp.zeros((b,), jnp.float32)))
    p_ref[0, 0, 0, :, :] = p
    s_ref[0, 0, 0, :] = s


def _fwd_kernel(idx_ref, tab_ref, x_ref, hin_ref, h_ref, *, c):
    def body(t, h):
        h = _matvec(tab_ref[idx_ref[t, 0], 0, :, :], h, False) + x_ref[t, 0, 0, :]
        h_ref[t, 0, 0, :] = h
        return h
    jax.lax.fori_loop(0, c, body, hin_ref[0, 0, 0, :])


def _bwd_kernel(idx_ref, slot_ref, tab_ref, g_ref, hp_ref, uin_ref, dtab_in, dx_ref, dw_ref, dtab_ref, *, c, k):
    del dtab_in

    def body(i, u):
        t = c - 1 - i
        a = tab_ref[idx_ref[t, 0], 0, :, :]
        lam = g_ref[t, 0, 0, :] + u
        dx_ref[t, 0, 0, :] = lam
        da = lam[:, None] * hp_ref[t, 0, 0, :][None, :]
        plgpu.atomic_add(dtab_ref, (idx_ref[t, 0], jnp.zeros((), jnp.int32), slice(None), slice(None)), da)
        for q in range(k):
            dw_ref[t, 0, 0, q] = jnp.sum(da * tab_ref[slot_ref[t, 0, q], 0, :, :])
        return _matvec(a, lam, True)
    jax.lax.fori_loop(0, c, body, uin_ref[0, 0, 0, :])


def _specs(T, B, g, b, L, c):
    nc = T // c
    idx = pl.BlockSpec((c, 1), lambda i, j, k: (k, i))
    tab = pl.BlockSpec((L, 1, b, b), lambda i, j, k: (0, j, 0, 0))
    vec = pl.BlockSpec((c, 1, 1, b), lambda i, j, k: (k, i, j, 0))
    cvec = pl.BlockSpec((1, 1, 1, b), lambda i, j, k: (k, i, j, 0))
    cmat = pl.BlockSpec((1, 1, 1, b, b), lambda i, j, k: (k, i, j, 0, 0))
    return (B, g, nc), idx, tab, vec, cvec, cmat


def _summary(tab, idx, x, rev, c, interpret):
    T, B, g, b = x.shape
    grid, si, st, sv, scv, scm = _specs(T, B, g, b, tab.shape[0], c)
    nc = T // c
    return pl.pallas_call(functools.partial(_summary_kernel, c=c, rev=rev), grid=grid, in_specs=[si, st, sv],
                          out_specs=[scm, scv],
                          out_shape=[jax.ShapeDtypeStruct((nc, B, g, b, b), jnp.float32),
                                     jax.ShapeDtypeStruct((nc, B, g, b), jnp.float32)],
                          interpret=interpret)(idx, tab, x)


def _forward(tab, idx, x, c, interpret):
    T, B, g, b = x.shape
    p, s = _summary(tab, idx, x, False, c, interpret)
    end = jax.lax.associative_scan(_comb, (p, s), axis=0)[1]
    hin = jnp.concatenate([jnp.zeros_like(end[:1]), end[:-1]])
    grid, si, st, sv, scv, _ = _specs(T, B, g, b, tab.shape[0], c)
    return pl.pallas_call(functools.partial(_fwd_kernel, c=c), grid=grid, in_specs=[si, st, sv, scv], out_specs=sv,
                          out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32), interpret=interpret)(idx, tab, x, hin)


def _backward(tab, idx, slots, h, gh, c, interpret):
    T, B, g, b = h.shape
    K = slots.shape[-1]
    p, s = _summary(tab, idx, gh, True, c, interpret)
    mu = jax.lax.associative_scan(_comb, (p[::-1], s[::-1]), axis=0)[1][::-1]  # u at each chunk start
    uin = jnp.concatenate([mu[1:], jnp.zeros_like(mu[:1])])
    hp = jnp.concatenate([jnp.zeros_like(h[:1]), h[:-1]])
    grid, si, st, sv, scv, _ = _specs(T, B, g, b, tab.shape[0], c)
    ss = pl.BlockSpec((c, 1, K), lambda i, j, k: (k, i, 0))
    sw = pl.BlockSpec((c, 1, 1, K), lambda i, j, k: (k, i, j, 0))
    dx, dw, dtab = pl.pallas_call(
        functools.partial(_bwd_kernel, c=c, k=K), grid=grid, in_specs=[si, ss, st, sv, sv, scv, st],
        out_specs=[sv, sw, st],
        out_shape=[jax.ShapeDtypeStruct(h.shape, jnp.float32), jax.ShapeDtypeStruct((T, B, g, K), jnp.float32),
                   jax.ShapeDtypeStruct(tab.shape, jnp.float32)],
        input_output_aliases={6: 2}, interpret=interpret)(idx, slots, tab, gh, hp, uin, jnp.zeros_like(tab))
    return dx, dw.sum(2), dtab


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def leaf_scan(tab, slots, w, x, chunk=CHUNK, interpret=False):
    """tab (L, g, b, b), slots (T, B, K) leaf ids with slot 0 the walked leaf, w (T, B, K) straight-through weights,
    x (T, B, g, b) -> h (T, B, g, b), h_0 = 0. The value uses A_{slot 0}; w only carries the gradient."""
    return _forward(tab, slots[..., 0], x, chunk, interpret)


def _leaf_scan_fwd(tab, slots, w, x, chunk, interpret):
    h = _forward(tab, slots[..., 0], x, chunk, interpret)
    return h, (tab, slots, w, h)


def _leaf_scan_bwd(chunk, interpret, res, gh):
    tab, slots, w, h = res
    dx, dw, dtab = _backward(tab, slots[..., 0], slots, h, gh, chunk, interpret)
    return dtab, None, dw.astype(w.dtype), dx


leaf_scan.defvjp(_leaf_scan_fwd, _leaf_scan_bwd)


def token_scan(a, x, chunk=CHUNK, interpret=False):
    """Per-token transitions a (T, B, g, b, b): the table is a itself, indexed by (t, b)."""
    T, B = x.shape[:2]
    ids = (jnp.arange(T)[:, None] * B + jnp.arange(B)[None, :]).astype(jnp.int32)[..., None]
    return leaf_scan(a.reshape((T * B,) + a.shape[2:]), ids, jnp.ones(ids.shape, x.dtype), x, chunk, interpret)
