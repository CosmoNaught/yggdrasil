"""Top-k descent of the binary tree router of a routed layer."""
import jax
import jax.numpy as jnp

GREEDY_BONUS = 1e9  # added to the greedy child's score so top_k keeps it, in slot 0


def _straight_through(w):
    """Forward: one-hot on slot 0. Backward: gradient of the softmax weights w."""
    slot0 = jax.nn.one_hot(jnp.zeros(w.shape[:-1], jnp.int32), w.shape[-1], dtype=w.dtype)
    return slot0 + w - jax.lax.stop_gradient(w)


def route(p, f, depth, topk, temperature, mode, key=None, noise=0.0):
    """f (T, B, F) -> expert ids (T, B, k), weights (T, B, k); slot 0 is the greedy path.

    mode "hard": greedy path only (k = 1), weight 1. mode "st": top-k paths, straight-through weights.
    """
    topk = 1 if mode == "hard" else topk
    node = jnp.zeros(f.shape[:2] + (1,), jnp.int32)
    lp = jnp.zeros(f.shape[:2] + (1,), f.dtype)
    for level in range(depth):
        s = jnp.einsum("tbf,tbqf->tbq", f, p["router_weight"][node]) + p["router_bias"][node]
        if key is not None and noise > 0:
            s = s + noise * jax.random.logistic(jax.random.fold_in(key, level), s.shape)
        cand_lp = jnp.stack([lp + jax.nn.log_sigmoid(-s / temperature), lp + jax.nn.log_sigmoid(s / temperature)], -1)
        cand_lp = cand_lp.reshape(s.shape[:-1] + (-1,))
        cand = jnp.stack([2 * node + 1, 2 * node + 2], -1).reshape(cand_lp.shape)
        greedy = jax.nn.one_hot((s[..., 0] > 0).astype(jnp.int32), cand.shape[-1], dtype=f.dtype)
        _, sel = jax.lax.top_k(jax.lax.stop_gradient(cand_lp) + GREEDY_BONUS * greedy, min(cand.shape[-1], topk))
        node, lp = jnp.take_along_axis(cand, sel, -1), jnp.take_along_axis(cand_lp, sel, -1)
    expert = node - (2**depth - 1)
    if mode == "hard":
        return expert, jnp.ones_like(lp)
    return expert, _straight_through(jax.nn.softmax(lp, -1))
