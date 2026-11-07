"""Associative scan of the block-diagonal linear recurrence."""
import jax
import jax.numpy as jnp


def _combine(l, r):
    return (jnp.einsum("...ij,...jk->...ik", r[0], l[0]),
            jnp.einsum("...ij,...j->...i", r[0], l[1]) + r[1])


def scan(a, x, h0=None):
    """h_t = a_t h_{t-1} + x_t: a (T, B, g, b, b), x (T, B, g, b), h0 (B, g * b) or None (zeros) -> h (T, B, g, b)."""
    if h0 is not None:
        x = x.at[0].add(jnp.einsum("b...ij,b...j->b...i", a[0], h0.reshape(x.shape[1:])))
    return jax.lax.associative_scan(_combine, (a, x), axis=0)[1]
