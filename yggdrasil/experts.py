"""Stored transitions and the hierarchical residual experts (HRE) that hold the routed experts in training."""
import jax
import jax.numpy as jnp

from .config import EXPERT_KEYS


def transitions(p):
    """A_r = diag(clip(decay_r)) blockdiag(expm((S_r - S_r^T) / 2)) -> (E, d_state / b, b, b), b = block_size."""
    gen = 0.5 * (p["rotation"] - jnp.swapaxes(p["rotation"], -1, -2))
    return jnp.clip(p["decay"], -1.0, 1.0).reshape(gen.shape[:-1])[..., None] * jax.scipy.linalg.expm(gen)


def to_hre(p, depth):
    """Per-expert arrays (2^depth, ...) -> hre_levels [(2^d, ...) for d = 0..depth]; depth 0 holds expert 0 and
    deeper residuals start at zero, so every expert starts equal to expert 0."""
    scale = (depth + 1) ** -0.5
    hre = {k: [p[k][:1] / scale] + [jnp.zeros((2**d,) + p[k].shape[1:], p[k].dtype) for d in range(1, depth + 1)]
           for k in EXPERT_KEYS if k in p}
    return dict({k: v for k, v in p.items() if k not in hre}, hre_levels=hre)


@jax.custom_vjp
def hre_sum(residuals):
    """Expert j = (depth + 1)^-1/2 sum_d residuals[d][j >> (depth - d)] -> (2^depth, ...)."""
    acc = residuals[0]
    for r in residuals[1:]:
        acc = jnp.repeat(acc, 2, axis=0) + r
    return acc * (len(residuals) ** -0.5)


def _hre_sum_fwd(residuals):
    return hre_sum(residuals), None


def _hre_sum_bwd(_, g):  # one level at a time, so each expert gradient is read once
    depth = g.shape[0].bit_length() - 1
    g = g * ((depth + 1) ** -0.5)
    out = [g]
    for _ in range(depth):
        g = g.reshape((g.shape[0] // 2, 2) + g.shape[1:]).sum(1)
        out.append(g)
    return (out[::-1],)


hre_sum.defvjp(_hre_sum_fwd, _hre_sum_bwd)


def expert_bank(p):
    """Layer with hre_levels -> per-expert arrays; a layer without them is returned as is."""
    if "hre_levels" not in p:
        return p
    rest = {k: v for k, v in p.items() if k != "hre_levels"}
    return dict(rest, **{k: hre_sum(list(v)) for k, v in p["hre_levels"].items()})


def materialize(params):
    """Params in training form -> per-expert arrays, as checkpoints store them."""
    return dict(params, routed_layers=[expert_bank(p) for p in params["routed_layers"]])
