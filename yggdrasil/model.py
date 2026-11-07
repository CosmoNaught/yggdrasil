"""Yggdrasil: parameters, forward pass and checkpoints.

    h^l_t  = A_r h^l_{t-1} + U_r v(x_t) + c_r + U'_r V' rms(h^{l-1}_t)
    logits = output_bias + sum_l Q^l P^l_r h^l_t

The token layer (l = 0) uses expert r = x_t; routed layer l uses the expert its tree router picks from
[router_embed(x_t); rms(h^{l-1}_t)].
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from .config import RMS_EPS, upgrade_keys
from .experts import expert_bank, transitions
from .router import route
from .scan import scan

# in case jax was imported before the package set the variable
jax.config.update("jax_default_matmul_precision", os.environ["JAX_DEFAULT_MATMUL_PRECISION"])

MAX_TABLE_ELEMS = 2**28  # largest table U_r v(x) + c_r (experts x vocab x d_state) precomputed per step

# Init scales (std of a normal draw unless noted)
TOKEN_EMBED_STD = 1.0
ROUTER_EMBED_STD = 1.0
INPUT_PROJ_GAIN = 0.5  # times input_rank^-1/2
BIAS_STD = 0.1
UNEMBED_GAIN = 0.1  # times readout_rank^-1/2
DECAY_INIT = (0.6, 1.0)  # uniform range
KEYS_PER_LAYER = 16  # split budget; changing it changes every draw


def _normal(key, shape, scale):
    return scale * jax.random.normal(key, shape)


def _rotation_init(key, n_experts, cfg):
    """Block stds rise geometrically from rotation_init_std (first block) to 1 (last block)."""
    s0, g = cfg.rotation_init_std, cfg.d_state // cfg.block_size
    stds = s0 * (1.0 / s0) ** (jnp.arange(g) / max(1, g - 1))
    return _normal(key, (n_experts, g, cfg.block_size, cfg.block_size), s0) * (stds / s0)[:, None, None]


def _decay_init(key, n_experts, cfg):
    d = jax.random.uniform(key, (n_experts, cfg.d_state), minval=DECAY_INIT[0], maxval=DECAY_INIT[1])
    return d.at[:, :cfg.block_size].set(0.0)  # the first block starts memoryless: a function of the current input


def _layer_init(keys, i, cfg, vocab, classes):
    routed = i > 0
    d, rank, out_rank = cfg.d_state, cfg.input_rank, cfg.readout_rank
    n_experts = 2**cfg.router_depth if routed else vocab
    router_key = next(keys)  # drawn by every layer, so the key sequence does not depend on the layer kind
    p = {"token_embed": _normal(next(keys), (vocab, rank), TOKEN_EMBED_STD),
         "input_proj": _normal(next(keys), (n_experts, d, rank), INPUT_PROJ_GAIN * rank**-0.5),
         "bias": _normal(next(keys), (n_experts, d), BIAS_STD),
         "readout": _normal(next(keys), (n_experts, out_rank, d), d**-0.5),
         "unembed": _normal(next(keys), (classes, out_rank), UNEMBED_GAIN * out_rank**-0.5)}
    if routed:
        n_in = cfg.router_token_dim + (0 if cfg.router_input == "token" else d)
        n_nodes = 2**cfg.router_depth - 1
        p["router_embed"] = _normal(router_key, (vocab, cfg.router_token_dim), ROUTER_EMBED_STD)
        p["router_weight"] = _normal(next(keys), (n_nodes, n_in), n_in**-0.5)
        p["router_bias"] = jnp.zeros((n_nodes,))
    rotation_key, decay_key = next(keys), next(keys)
    if _has_transition(i, cfg):
        p["rotation"] = _rotation_init(rotation_key, n_experts, cfg)
        p["decay"] = _decay_init(decay_key, n_experts, cfg)
    if _has_layer_input(i, cfg):
        p["layer_input_down"] = _normal(next(keys), (cfg.layer_input_rank, d), d**-0.5)
        p["layer_input_proj"] = jnp.zeros((n_experts, d, cfg.layer_input_rank))
    return p


def _has_transition(i, cfg):
    return i == 0 or cfg.expert_type == "transition"


def _has_layer_input(i, cfg):
    return i > 0 and cfg.layer_input_rank > 0


def init(key, cfg, vocab, classes):
    """-> {"token_layer", "routed_layers": [...] * (n_layers - 1), "output_bias"}; keys in config.PARAMS."""
    keys = iter(jax.random.split(key, KEYS_PER_LAYER * cfg.n_layers + 2))
    layers = [_layer_init(keys, i, cfg, vocab, classes) for i in range(cfg.n_layers)]
    return {"token_layer": layers[0], "routed_layers": layers[1:], "output_bias": jnp.zeros((classes,))}


def rms(h):
    return h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + RMS_EPS)


def _drive(p, expert, w, xs):
    """sum_q w_q (U_r v(x) + c_r) at experts r = expert[..., q] -> (T, B, d_state)."""
    n_experts, d = p["bias"].shape
    vocab = p["token_embed"].shape[0]
    if n_experts * vocab * d <= MAX_TABLE_ELEMS:
        table = jnp.einsum("lkr,vr->lvk", p["input_proj"], p["token_embed"]) + p["bias"][:, None, :]
        return jnp.einsum("tbq,tbqk->tbk", w, table.reshape(-1, d)[expert * vocab + xs[:, :, None]])
    return jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["input_proj"][expert], p["token_embed"][xs]) + \
        jnp.einsum("tbq,tbqk->tbk", w, p["bias"][expert])


def layer(p, expert, w, xs, below, h0=None, transition=True):
    """expert, w (T, B, k), xs (T, B), below (T, B, d_state) or None, h0 (B, d_state) or None -> (T, B, d_state).

    transition False: A_r = 0 (expert_type "output").
    """
    drive = _drive(p, expert, w, xs)
    if below is not None:
        down = below @ p["layer_input_down"].T
        drive = drive + jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["layer_input_proj"][expert], down)
    if not transition:
        return drive
    abar = jnp.einsum("tbq,tbq...->tb...", w, transitions(p)[expert])
    return scan(abar, drive.reshape(abar.shape[:-1]), h0).reshape(drive.shape)


def _routed_experts(p, i, xs, below, cfg, mode, key):
    """Router of routed layer i -> expert ids, weights (T, B, k)."""
    f = p["router_embed"][xs]
    if cfg.router_input == "token_state":
        f = jnp.concatenate([f, below], -1)
    layer_key = None if key is None else jax.random.fold_in(jax.random.fold_in(key, i), 1)
    return route(p, f, cfg.router_depth, cfg.router_topk, cfg.router_temperature, mode, layer_key, cfg.router_noise)


def forward(params, tokens, cfg, mode="hard", key=None, state=None):
    """tokens (B, T) -> log-probs (B, T, classes), last states [(B, d_state)] per layer.

    mode "hard": greedy routing (evaluation). "st": straight-through top-k routing with router noise (training, key).
    state: initial states [(B, d_state)] per layer, or None for zeros.
    """
    xs = jnp.swapaxes(tokens, 0, 1)
    layers = [params["token_layer"]] + [expert_bank(p) for p in params["routed_layers"]]
    below, logits, last = None, params["output_bias"], []
    for i, p in enumerate(layers):
        if i == 0:
            expert, w = xs[:, :, None], jnp.ones(xs.shape + (1,), jnp.float32)
        else:
            expert, w = _routed_experts(p, i, xs, below, cfg, mode, key)
        layer_input = below if _has_layer_input(i, cfg) else None
        h0 = None if state is None else state[i]
        h = layer(p, expert, w, xs, layer_input, h0, _has_transition(i, cfg))
        last.append(h[-1])
        below = rms(h)
        logits = logits + jnp.einsum("tbq,tbqrk,tbk->tbr", w, p["readout"][expert], h) @ p["unembed"].T
    return jnp.swapaxes(jax.nn.log_softmax(logits, -1), 0, 1), last


def save(path, params):
    """Flat .npz with keys token_layer/<key>, routed_layers/<i>/<key>, output_bias."""
    flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in kp): np.asarray(v)
            for kp, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, **flat)


def load(path):
    """.npz from save(), with new or released (old) keys -> params with new keys."""
    with np.load(path) as z:
        flat = upgrade_keys({name: z[name] for name in z.files})
    params = {"token_layer": {}, "routed_layers": [], "output_bias": None}
    for name, value in flat.items():
        top, *rest = name.split("/")
        if top == "token_layer":
            params[top][rest[0]] = value
        elif top == "routed_layers":
            i = int(rest[0])
            params[top] += [{} for _ in range(i + 1 - len(params[top]))]
            params[top][i][rest[1]] = value
        elif top == "output_bias":
            params[top] = value
        else:
            raise ValueError(f"{path}: unknown checkpoint key {name!r}")
    return jax.tree_util.tree_map(jnp.asarray, params)
