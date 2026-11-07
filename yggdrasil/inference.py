"""Yggdrasil inference: float32, hard routing as at evaluation, one token per stream per step.

    model = Yggdrasil.load("results/<run>.npz")      # backend="triton": the fused GPU step (inference_triton.py)
    state = model.init_state(batch_size)
    logits, state = model.step(tokens, state)       # tokens (B,) -> logits (B, classes)
    bits = model.score(windows)                     # (B, L) -> bits of windows[:, 1:] per window (B,)
    text = model.generate(prompt, 200, temperature=0.8, top_k=20)

The token layer uses expert r = x. Each routed layer sends f = [router_embed(x); rms(h_below)] down a binary tree
router of depth D, right at node j iff f . W_j + b_j > 0, and uses the expert r its path ends at (model.py notation):

    h      <- A_r h + U_r v(x) + c_r + U'_r V' rms(h_below)
    A_r     = diag(clip(d_r, -1, 1)) blockdiag(expm((G_r - G_r^T) / 2))
    logits  = output_bias + sum_l Q^l P^l_r h^l
"""

import numpy as np

from tasks.text import LN2

from .config import upgrade_keys

try:  # backend="triton" needs only torch, so jax and the jax modules of the package are optional here
    import jax
    import jax.numpy as jnp

    from .experts import transitions
    from .model import rms
except ImportError:
    jax = jnp = None
_jit = jax.jit if jax else lambda f: f

# Per-layer checkpoint keys this module reads; any other key is an unsupported model variant.
LAYER_KEYS = {"token_embed", "input_proj", "bias", "rotation", "decay", "readout", "unembed",
              "router_embed", "router_weight", "router_bias", "layer_input_down", "layer_input_proj"}


def router_depth(p):
    return (p["router_weight"].shape[0] + 1).bit_length() - 1


def route(p, x, below):
    """Expert (B,) of tokens x (B,) given rms(h_below) (B, n)."""
    f = p["router_embed"][x]
    if p["router_weight"].shape[1] > f.shape[1]:  # routes on the state below too
        f = jnp.concatenate([f, below], -1)
    depth = router_depth(p)
    node = jnp.zeros(x.shape, jnp.int32)
    for _ in range(depth):
        node = 2 * node + 1 + (jnp.einsum("bf,bf->b", f, p["router_weight"][node]) + p["router_bias"][node] > 0)
    return node - (2**depth - 1)


def update(p, r, x, h, below):
    """New state (B, n) and readout P_r h (B, readout_rank) of one layer at experts r."""
    h_new = p["table"][r, x]
    if "layer_input_down" in p:
        h_new = h_new + jnp.einsum("bkr,br->bk", p["layer_input_proj"][r], below @ p["layer_input_down"].T)
    if "transition" in p:
        a = p["transition"][r]
        h_new = h_new + jnp.einsum("bgij,bgj->bgi", a, h.reshape(a.shape[:-1])).reshape(h.shape)
    return h_new, jnp.einsum("brk,bk->br", p["readout"][r], h_new)


def step(params, x, state):
    """x (B,) int, state (layers, B, n) -> logits (B, V), state."""
    with jax.default_matmul_precision("highest"):
        logits, below, new = params["output_bias"], None, []
        for i, p in enumerate(params["layers"]):
            h, z = update(p, x if i == 0 else route(p, x, below), x, state[i], below)
            new.append(h)
            below = rms(h)
            logits = logits + z @ p["unembed"].T
        return logits, jnp.stack(new)


@_jit
def _step(params, x, state):
    return step(params, x, state)


@_jit
def _bits(params, tokens, state):
    """Bits of tokens[:, 1:] given tokens[:, :-1], per row (B,), and the state after tokens[:, :-1]."""
    def f(st, xy):
        logits, st = step(params, xy[0], st)
        lp = jax.nn.log_softmax(logits, -1)
        return st, -jnp.take_along_axis(lp, xy[1][:, None], -1)[:, 0]
    state, nll = jax.lax.scan(f, state, (tokens[:, :-1].T, tokens[:, 1:].T))
    return nll.sum(0) / LN2, state


@_jit
def _feed(params, tokens, state):
    """Logits after the last token of tokens (B, L), and the state."""
    def f(carry, x):
        return step(params, x, carry[1]), None
    (logits, state), _ = jax.lax.scan(f, (params["output_bias"][None].repeat(len(tokens), 0), state), tokens.T)
    return logits, state


def _sample(logits, key, temperature, top_k):
    if temperature == 0:
        return logits.argmax(-1).astype(jnp.int32)
    if top_k:
        logits = jnp.where(logits < jax.lax.top_k(logits, top_k)[0][:, -1:], -jnp.inf, logits)
    return jax.random.categorical(key, logits / temperature).astype(jnp.int32)


def _nest(z):
    """Flat .npz with "/"-joined key paths -> nested dicts, with lists where every key is an index."""
    out = {}
    for name in z.files:
        *path, key = name.split("/")
        d = out
        for k in path:
            d = d.setdefault(k, {})
        d[key] = z[name].astype(np.float32)

    def lists(d):
        if not isinstance(d, dict):
            return d
        if d and all(k.isdigit() for k in d):
            return [lists(d[str(i)]) for i in range(len(d))]
        return {k: lists(v) for k, v in d.items()}
    return lists(out)


def read(path):
    """Checkpoint (.npz, old or new keys) -> layers [token layer, routed layers...] as dicts of float32 arrays, and
    the output bias (V,)."""
    params = upgrade_keys(_nest(np.load(path)))
    layers = [params["token_layer"], *params["routed_layers"]]
    for p in layers:
        if bad := set(p) - LAYER_KEYS:
            raise NotImplementedError(f"{path}: unsupported layer arrays {sorted(bad)}")
        if len(p["readout"]) != len(p["bias"]):
            raise NotImplementedError(f"{path}: the readout needs one matrix per expert")
    return layers, params["output_bias"]


class Yggdrasil:
    def __init__(self, layers, output_bias):
        """layers, output_bias as from read(). Precomputes T_r[x] = U_r v(x) + c_r and the transitions A_r."""
        self.params = {"layers": [], "output_bias": jnp.asarray(output_bias)}
        with jax.default_matmul_precision("highest"):
            for p in layers:
                q = {k: jnp.asarray(v) for k, v in p.items()
                     if k not in ("input_proj", "token_embed", "bias", "rotation", "decay")}
                q["table"] = jnp.einsum("lkr,vr->lvk", p["input_proj"], p["token_embed"]) + p["bias"][:, None]
                if "rotation" in p:  # absent when the experts store outputs only
                    q["transition"] = transitions({k: jnp.asarray(p[k]) for k in ("rotation", "decay")})
                self.params["layers"].append(q)
        self.layers, self.n = len(layers), layers[0]["bias"].shape[1]
        self.vocab, self.classes = layers[0]["token_embed"].shape[0], len(output_bias)
        self.depth = router_depth(layers[1]) if len(layers) > 1 else 0

    @classmethod
    def load(cls, path, backend="jax", **kw):
        """From a training checkpoint (.npz). backend "triton": the fused GPU step, kw device="cuda", cuda_graph."""
        if backend == "triton":
            try:
                from .inference_triton import TritonYggdrasil
            except ImportError as e:
                raise ImportError(f"backend='triton' needs torch and triton: pip install 'yggdrasil[gpu]' ({e})") from e
            return TritonYggdrasil(*read(path), **kw)
        if backend != "jax":
            raise ValueError(f"backend {backend!r}: jax or triton")
        return cls(*read(path), **kw)

    def init_state(self, batch_size):
        """Zero state of batch_size streams, (layers, B, n)."""
        return jnp.zeros((self.layers, batch_size, self.n), jnp.float32)

    def step(self, tokens, state):
        """One token per stream: tokens (B,) -> logits (B, classes) of the next token, state."""
        return _step(self.params, jnp.asarray(tokens, jnp.int32), state)

    def score(self, tokens, state=None, batch=1024):
        """Bits of tokens[:, 1:] given the tokens before them, per row of tokens (B, L), as float64 (B,).
        Rows start from state, or from zero; bits per token = score(tokens).sum() / (B (L - 1))."""
        tokens = np.atleast_2d(np.asarray(tokens, np.int32))
        out = []
        for i in range(0, len(tokens), batch):
            x = tokens[i:i + batch]
            st = self.init_state(len(x)) if state is None else state[:, i:i + batch]
            out.append(np.asarray(_bits(self.params, jnp.asarray(x), st)[0], np.float64))
        return np.concatenate(out)

    def generate(self, prompt, n, temperature=1.0, top_k=0, seed=0):
        """Prompt (L,) or (B, L), L >= 1, followed by n sampled tokens. temperature 0: greedy."""
        prompt = np.asarray(prompt, np.int32)
        x = np.atleast_2d(prompt)
        logits, state = _feed(self.params, jnp.asarray(x), self.init_state(len(x)))
        key, out = jax.random.PRNGKey(seed), []
        for i in range(n):
            tok = _sample(logits, jax.random.fold_in(key, i), temperature, top_k)
            out.append(tok)
            if i + 1 < n:
                logits, state = self.step(tok, state)
        out = np.concatenate([x, np.stack(out, 1)], 1) if n else x
        return out[0] if prompt.ndim == 1 else out
