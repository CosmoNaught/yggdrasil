"""jax.scipy.linalg.expm of fp32 (..., 16, 16) blocks on a GPU, one Pallas kernel (Triton backend) per block.

Forward, as jax.scipy.linalg.expm in fp32: L1 norm; s = max(0, floor(log2(|A|_1 / 3.9257))) squarings; Pade order
3, 5 or 7 chosen from |A|_1 (thresholds 0.4259 and 1.8802) and applied to A 2^-s; R = Q^-1 P by Gaussian elimination
with partial pivoting; s squarings; NaN when s > 16. The products are exact fp32 (IEEE) dots.

Backward: the approximant f(A) = r_m(A 2^-s)^(2^s) is a rational function with real coefficients, so the adjoint of
its derivative at A is its derivative at A^T: dL/dA = L_f(A^T, G), with m and s those of the forward. The kernel runs
that directional derivative forward through the Pade terms, the solve (dR = Q^-1 (dP - dQ R)) and the squarings
(dX <- X dX + dX X), so nothing is stored between the passes. The values equal the autodiff gradient of
jax.scipy.linalg.expm up to floating-point rounding.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_HI = jax.lax.Precision.HIGHEST
MAXNORM = 3.925724783138660
CONDS = (4.258730016922831e-001, 1.880152677804762e+000)
MAX_SQ = 16
B3 = (120., 60., 12., 1.)
B5 = (30240., 15120., 3360., 420., 30., 1.)
B7 = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
WARPS = (1, 2)  # forward, backward: one 16 x 16 block per program


def _mm(a, c):
    return jnp.dot(a, c, precision=_HI, preferred_element_type=jnp.float32)


def _sel(idx, a, b, c):
    return jnp.where(idx == 0, a, jnp.where(idx == 1, b, c))


def _norm_steps(a):
    """-> squarings s (int32), 2^-s, Pade index (0: order 3, 1: 5, 2: 7) from the L1 norm (max column sum)."""
    l1 = jnp.max(jnp.sum(jnp.abs(a), axis=0))
    s = jnp.maximum(0.0, jnp.floor(jnp.log2(l1 / MAXNORM)))
    idx = (l1 >= CONDS[0]).astype(jnp.int32) + (l1 >= CONDS[1]).astype(jnp.int32)
    return s.astype(jnp.int32), s, idx


def _pade(x, eye, idx):
    """U, V of the selected Pade order, in jax.scipy's operation order."""
    x2 = _mm(x, x)
    x4 = _mm(x2, x2)
    x6 = _mm(x4, x2)
    inner = _sel(idx, B3[3] * x2 + B3[1] * eye, B5[5] * x4 + B5[3] * x2 + B5[1] * eye,
                 B7[7] * x6 + B7[5] * x4 + B7[3] * x2 + B7[1] * eye)
    u = _mm(x, inner)
    v = _sel(idx, B3[2] * x2 + B3[0] * eye, B5[4] * x4 + B5[2] * x2 + B5[0] * eye,
             B7[6] * x6 + B7[4] * x4 + B7[2] * x2 + B7[0] * eye)
    return u, v, (x2, x4, x6, inner)


def _pade_jvp(x, e, eye, idx, parts):
    x2, x4, x6, inner = parts
    d2 = _mm(e, x) + _mm(x, e)
    d4 = _mm(d2, x2) + _mm(x2, d2)
    d6 = _mm(d4, x2) + _mm(x4, d2)
    dinner = _sel(idx, B3[3] * d2, B5[5] * d4 + B5[3] * d2, B7[7] * d6 + B7[5] * d4 + B7[3] * d2)
    du = _mm(e, inner) + _mm(x, dinner)
    dv = _sel(idx, B3[2] * d2, B5[4] * d4 + B5[2] * d2, B7[6] * d6 + B7[4] * d4 + B7[2] * d2)
    return du, dv


def _solve(q, p, rows, cols):
    """Q^-1 P (16 x 16 each) by Gaussian elimination with partial pivoting, then back substitution."""
    b = q.shape[0]
    pick_col = lambda m, k: jnp.sum(jnp.where(cols == k, m, 0.0), axis=1)  # (b,)
    pick_row = lambda m, k: jnp.sum(jnp.where(rows == k, m, 0.0), axis=0)  # (b,)
    r1 = jax.lax.broadcasted_iota(jnp.int32, (b,), 0)
    for k in range(b):
        col = jnp.abs(pick_col(q, k))
        cand = jnp.where(r1 >= k, col, -1.0)
        mx = jnp.max(cand)
        piv = jnp.min(jnp.where(cand == mx, r1, b))
        qk, qp, pk, pp = pick_row(q, k), pick_row(q, piv), pick_row(p, k), pick_row(p, piv)
        q = jnp.where(rows == k, qp[None, :], jnp.where(rows == piv, qk[None, :], q))
        p = jnp.where(rows == k, pp[None, :], jnp.where(rows == piv, pk[None, :], p))
        colk = pick_col(q, k)
        pv = jnp.sum(jnp.where(r1 == k, colk, 0.0))
        lk = jnp.where(r1 > k, colk / pv, 0.0)[:, None]
        q = q - lk * qp[None, :]
        p = p - lk * pp[None, :]
    x = jnp.zeros_like(p)
    for k in range(b - 1, -1, -1):
        qk, yk = pick_row(q, k), pick_row(p, k)
        ukk = jnp.sum(jnp.where(r1 == k, qk, 0.0))
        acc = jnp.sum(jnp.where(rows > k, qk[:, None], 0.0) * x, axis=0)  # sum_{j>k} U_kj x_j
        x = jnp.where(rows == k, ((yk - acc) / ukk)[None, :], x)
    return x


def _fwd_kernel(a_ref, r_ref):
    """r (1, b, b) = expm of one block a (1, b, b)."""
    a = a_ref[0, :, :]
    b = a.shape[0]
    rows = jax.lax.broadcasted_iota(jnp.int32, (b, b), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (b, b), 1)
    eye = (rows == cols).astype(jnp.float32)
    si, s, idx = _norm_steps(a)
    x = a / jnp.exp2(s)
    u, v, _ = _pade(x, eye, idx)
    r = _solve(-u + v, u + v, rows, cols)
    r = jax.lax.fori_loop(0, jnp.minimum(si, MAX_SQ), lambda _, r: _mm(r, r), r)
    r_ref[0, :, :] = jnp.where(si > MAX_SQ, jnp.nan, r)


def _bwd_kernel(a_ref, g_ref, d_ref):
    """d (1, b, b) = L_f(a^T, g): the gradient of <g, expm(a)> for one block, f with the forward's m and s."""
    a = a_ref[0, :, :]
    g = g_ref[0, :, :]
    b = a.shape[0]
    rows = jax.lax.broadcasted_iota(jnp.int32, (b, b), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (b, b), 1)
    eye = (rows == cols).astype(jnp.float32)
    si, s, idx = _norm_steps(a)  # m and s of the forward (from A, not A^T)
    sc = jnp.exp2(s)
    x, e = a.T / sc, g / sc
    u, v, parts = _pade(x, eye, idx)
    du, dv = _pade_jvp(x, e, eye, idx, parts)
    q = -u + v
    r = _solve(q, u + v, rows, cols)
    dr = _solve(q, (du + dv) - _mm(-du + dv, r), rows, cols)

    def sq(_, c):
        r, dr = c
        return _mm(r, r), _mm(r, dr) + _mm(dr, r)
    r, dr = jax.lax.fori_loop(0, jnp.minimum(si, MAX_SQ), sq, (r, dr))
    d_ref[0, :, :] = jnp.where(si > MAX_SQ, jnp.nan, dr)


def _blk(b):
    return pl.BlockSpec((1, b, b), lambda i: (i, 0, 0))


def _call(kernel, n_in, M, b, warps, *args):
    return pl.pallas_call(kernel, grid=(M,), in_specs=[_blk(b)] * n_in, out_specs=_blk(b),
                          out_shape=jax.ShapeDtypeStruct((M, b, b), jnp.float32),
                          compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1))(*args)


@jax.custom_vjp
def _expm_flat(a):
    M, b, _ = a.shape
    return _call(_fwd_kernel, 1, M, b, WARPS[0], a)


def _expm_flat_fwd(a):
    return _expm_flat(a), a


def _expm_flat_bwd(a, g):
    M, b, _ = a.shape
    return (_call(_bwd_kernel, 2, M, b, WARPS[1], a, g.astype(jnp.float32)),)


_expm_flat.defvjp(_expm_flat_fwd, _expm_flat_bwd)


def fits(a):
    return a.dtype == jnp.float32 and a.shape[-1] == a.shape[-2] == 16 and jax.default_backend() == "gpu"


def expm(a):
    """jax.scipy.linalg.expm of fp32 (..., 16, 16) on GPU; any other input goes to jax.scipy.linalg.expm."""
    if not fits(a):
        return jax.scipy.linalg.expm(a)
    return _expm_flat(a.reshape((-1,) + a.shape[-2:])).reshape(a.shape)
