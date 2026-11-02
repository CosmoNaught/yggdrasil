"""Yggdrasil: a recurrent layer that retrieves its transition instead of computing it.

Every layer l keeps a state h in R^n and a bank of leaves. Per token x_t:

    r      = x_t (layer 0)  or  Tree([e(x_t); nu(h^{l-1}_t)])   (depth-D walk, node j: right iff w_j . f + b_j > 0)
    h^l_t  = A_r h^l_{t-1} + T_r[x_t] (+ U_r W nu(h^{l-1}_t))     T_r[x] = U_r v(x) + c_r, one stored row per token
    A_r    = diag(clip(d_r, -1, 1)) blockdiag_g(expm(S_rg / 2 - S_rg^T / 2))      rotations times decays, block b
    logits = beta + sum_l Q^l P^l_r h^l_t

nu(h) = h / rms(h). The tree reads the layer below after it absorbed x_t, never the layer's own past, so within a layer
every transition is fixed by the input: each layer is one associative scan in training, and a step reads one leaf.
Training routes hard (one leaf, as at inference) with a straight-through beam gradient at temperature ROUTE_TAU, and
stores each branch leaf as a sum of increments along its tree path (to_tree), folded into plain leaf tables when saved.

    python models/yggdrasil/training/train.py train --task s5_word                      state tracking (s5_word a5_word parity_check
                                                             cycle_navigation modular_arithmetic)
    python models/yggdrasil/training/train.py train --task enwik8 --depth 12            byte-level language modelling (also: hg38, bits per base)
    python models/yggdrasil/training/train.py train --task human_nontata_promoters      Genomic Benchmarks classification (8 datasets)
    python models/yggdrasil/training/train.py train --task s5_word --seeds 160,161,162  one W&B sweep trial = several seeds, objective = mean
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[3]  # the repository (data/, results/)
LN2 = float(np.log(2.0))
SKEW_STD = 0.05  # skew std of the first block of a leaf; the last block has std 1
ROUTE_TAU = 0.5  # backward temperature of the straight-through routing


# ----------------------------------------------------------------------------------------------------------- model

def _normal(key, shape, scale):
    return scale * jax.random.normal(key, shape)


def init(key, vocab, classes, n=128, block=16, depth=8, layers=2, feat=16, rank=16, out_rank=8, lower_rank=16,
         shared_out=False, branch="dyn", context=1, bits=16):
    """{"root", "branches" (layers - 1), "head_b"}. Start: d ~ U[0.6, 1] with the first block of every leaf at d = 0
    (exact reset); skew std log-spaced 0.05 -> 1.0 over the blocks of a leaf.
    lower_rank: rank of the input from the layer below (0 = none). shared_out: one readout per layer, not per leaf.
    branch (ablations): "dyn" = the model; "out" = branch leaves store no transition (A_r = 0: the leaf's effect ends
    at the token, a stored output); "tok" = the branch tree reads the token only.
    context > 1: the root leaf is the last `context` tokens (context_leaf), min(vocab^context, 2^bits) leaves."""
    assert n % block == 0 and branch in ("dyn", "out", "tok") and (branch != "tok" or feat != n)
    assert context == 1 or vocab**context < 2**32
    g = n // block
    keys = iter(jax.random.split(key, 16 * layers + 2))
    parts = []
    for i in range(layers):
        leaves = (vocab if context == 1 else min(vocab**context, 2**bits)) if i == 0 else 2**depth
        emb = _normal(next(keys), (vocab, feat), 1.0)  # node features (drawn for the root too: fixed key order)
        p = {"v_tok": _normal(next(keys), (vocab, rank), 1.0),
             "u_in": _normal(next(keys), (leaves, n, rank), 0.5 * rank**-0.5),
             "c": _normal(next(keys), (leaves, n), 0.1),
             "p_out": _normal(next(keys), (leaves, out_rank, n), n**-0.5),
             "q_out": _normal(next(keys), (classes, out_rank), 0.1 * out_rank**-0.5)}
        if shared_out:
            p["p_out"] = p["p_out"][:1]
        if i > 0:
            n_in = feat if branch == "tok" else feat + n
            p["tok"] = emb
            p["w_node"] = _normal(next(keys), (2**depth - 1, n_in), n_in**-0.5)
            p["b_node"] = jnp.zeros((2**depth - 1,))
        stds = SKEW_STD * (1.0 / SKEW_STD) ** (jnp.arange(g) / max(1, g - 1))
        p["skew"] = _normal(next(keys), (leaves, g, block, block), SKEW_STD) * (stds / SKEW_STD)[:, None, None]
        p["d_raw"] = jax.random.uniform(next(keys), (leaves, n), minval=0.6, maxval=1.0).at[:, :block].set(0.0)
        if branch == "out" and i > 0:
            del p["skew"], p["d_raw"]
        if lower_rank and i > 0:
            p["v_low"] = _normal(next(keys), (lower_rank, n), n**-0.5)
            p["u_low"] = jnp.zeros((leaves, n, lower_rank))
        if context > 1 and i == 0:
            p["context"] = jnp.zeros((context, 0))  # the context length as a shape (no values, never trained)
        parts.append(p)
    return {"root": parts[0], "branches": parts[1:], "head_b": jnp.zeros((classes,))}


def context_leaf(xs, k, vocab, leaves, prev=None):
    """Root leaves (T, B) of the last k tokens of xs (T, B): their base-vocab number c (the current token is the last
    digit; tokens before the window come from prev (B, k - 1), else 0). Exact when vocab^k = leaves, else the top bits
    of c * 2654435761 mod 2^32 (leaves = 2^bits). A step rolls c = (c mod vocab^(k - 1)) vocab + x: O(1)."""
    pad = jnp.zeros((k - 1, xs.shape[1]), xs.dtype) if prev is None else jnp.swapaxes(prev, 0, 1).astype(xs.dtype)
    full = jnp.concatenate([pad, xs]).astype(jnp.uint32)
    c = jnp.zeros(xs.shape, jnp.uint32)
    for j in range(k):
        c = c * jnp.uint32(vocab) + full[j:j + xs.shape[0]]
    if vocab**k == leaves:
        return c.astype(jnp.int32)
    return ((c * jnp.uint32(2654435761)) >> jnp.uint32(32 - (leaves.bit_length() - 1))).astype(jnp.int32)


def _expm(sk, chunk=16384):
    """expm of (..., b, b); in checkpointed chunks when large (peak memory O(chunk))."""
    flat = sk.reshape((-1,) + sk.shape[-2:])
    if flat.shape[0] <= chunk:
        return jax.scipy.linalg.expm(sk)
    pad = -flat.shape[0] % chunk
    flat = jnp.concatenate([flat, jnp.zeros((pad,) + flat.shape[1:], flat.dtype)]) if pad else flat
    out = jax.lax.map(jax.checkpoint(jax.scipy.linalg.expm), flat.reshape((-1, chunk) + flat.shape[1:]))
    return out.reshape((-1,) + flat.shape[1:])[:flat.shape[0] - pad].reshape(sk.shape)


def transitions(p):
    """A_r as blocks (leaves, n / b, b, b)."""
    sk = 0.5 * (p["skew"] - jnp.swapaxes(p["skew"], -1, -2))
    return jnp.clip(p["d_raw"], -1.0, 1.0).reshape(sk.shape[:-1])[..., None] * _expm(sk)


def _scan(a, b):
    """h_t = a_t h_{t-1} + b_t, h_0 = 0: a (T, ..., b, b), b (T, ..., b)."""
    comb = lambda l, r: (jnp.einsum("...ij,...jk->...ik", r[0], l[0]),
                         jnp.einsum("...ij,...j->...i", r[0], l[1]) + r[1])
    return jax.lax.associative_scan(comb, (a, b), axis=0)[1]


def _rms(h):
    return h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + 1e-6)


def walk(p, f, beam, tau, mode, key=None, route_noise=0.0):
    """Beam walk: f (T, B, F) -> leaf ids (T, B, K), path weights (T, B, K). Per level, each kept path tests its node and
    the K best children survive; slot 0 always follows the greedy path (the one the CPU walks).
    hard: slot 0 only; st: slot 0 forward, gradient of the softmax over the beam's path log-probabilities at tau."""
    depth = (p["w_node"].shape[0] + 1).bit_length() - 1
    beam = 1 if mode == "hard" else beam
    node = jnp.zeros(f.shape[:2] + (1,), jnp.int32)
    lp = jnp.zeros(f.shape[:2] + (1,), f.dtype)
    for level in range(depth):
        s = jnp.einsum("tbf,tbqf->tbq", f, p["w_node"][node]) + p["b_node"][node]
        if key is not None and route_noise > 0:
            s = s + route_noise * jax.random.logistic(jax.random.fold_in(key, level), s.shape)
        cand_lp = jnp.stack([lp + jax.nn.log_sigmoid(-s / tau), lp + jax.nn.log_sigmoid(s / tau)], -1)
        cand_lp = cand_lp.reshape(s.shape[:-1] + (-1,))
        cand = jnp.stack([2 * node + 1, 2 * node + 2], -1).reshape(cand_lp.shape)
        greedy = jax.nn.one_hot((s[..., 0] > 0).astype(jnp.int32), cand.shape[-1], dtype=f.dtype)
        _, sel = jax.lax.top_k(jax.lax.stop_gradient(cand_lp) + 1e9 * greedy, min(cand.shape[-1], beam))
        node, lp = jnp.take_along_axis(cand, sel, -1), jnp.take_along_axis(cand_lp, sel, -1)
    leaf = node - (2**depth - 1)
    if mode == "hard":
        return leaf, jnp.ones_like(lp)
    w = jax.nn.softmax(lp, -1)
    return leaf, jax.nn.one_hot(jnp.zeros(lp.shape[:-1], jnp.int32), lp.shape[-1], dtype=w.dtype) + w - \
        jax.lax.stop_gradient(w)


def part(p, leaf, w, xs, low, noise, h0=None):
    """States (T, B, n) of one layer for leaf ids / weights (T, B, K); low = nu(state below) or None (root).
    h0: state (B, n) before the first token (None = zero)."""
    n_leaves, n = p["c"].shape
    vocab = p["v_tok"].shape[0]
    if n_leaves * vocab * n <= 2**28:  # the CPU table T_r[x] = U_r v(x) + c_r, gathered per token
        table = jnp.einsum("lkr,vr->lvk", p["u_in"], p["v_tok"]) + p["c"][:, None, :]
        drive = jnp.einsum("tbq,tbqk->tbk", w, table.reshape(-1, n)[leaf * vocab + xs[:, :, None]])
    else:  # same values without building the table
        drive = jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["u_in"][leaf], p["v_tok"][xs]) + \
            jnp.einsum("tbq,tbqk->tbk", w, p["c"][leaf])
    if "v_low" in p:
        drive = drive + jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["u_low"][leaf], low @ p["v_low"].T)
    if noise is not None:
        drive = drive + noise
    if "skew" not in p:  # branch "out": no stored transition, the state is the drive alone
        return drive
    abar = jnp.einsum("tbq,tbq...->tb...", w, transitions(p)[leaf])
    shape, drive = drive.shape, drive.reshape(abar.shape[:-1])
    if h0 is not None:  # h_1 = A_1 h0 + b_1: fold the carried state into the first drive
        drive = drive.at[0].add(jnp.einsum("b...ij,b...j->b...i", abar[0], h0.reshape(drive.shape[1:])))
    return _scan(abar, drive).reshape(shape)


def forward(params, tokens, beam=8, tau=ROUTE_TAU, mode="hard", key=None, route_noise=0.0, noise_std=0.0, state=None,
            return_state=False, prev=None):
    """tokens (B, T) int -> log-probs (B, T, classes). mode: "hard" = the CPU path (one leaf per token and layer),
    "st" = hard forward with beam gradient (training). key: training noise (None = none).
    state: list of per-layer states (B, n) before the first token (None = zeros); return_state: also return the
    per-layer states after the last token (streaming evaluation: carry them into the next chunk).
    prev: the k - 1 tokens (B, k - 1) before the window for a context root (None = zeros)."""
    xs = jnp.swapaxes(tokens, 0, 1)
    parts = [params["root"]] + [leaf_table(p) for p in params["branches"]]
    low, logits, last = None, params["head_b"], []
    for i, p in enumerate(parts):
        pkey = None if key is None else jax.random.fold_in(key, i)
        noise = None
        if pkey is not None and noise_std > 0:
            noise = noise_std * jax.random.normal(jax.random.fold_in(pkey, 2), xs.shape + p["c"].shape[-1:])
        if i == 0:  # root: leaf = token (or the last k tokens: context_leaf), weight 1
            leaf = xs if "context" not in p else \
                context_leaf(xs, p["context"].shape[0], p["v_tok"].shape[0], p["c"].shape[0], prev)
            leaf, w = leaf[:, :, None], jnp.ones(xs.shape + (1,), jnp.float32)
        else:
            f = p["tok"][xs] if p["w_node"].shape[1] == p["tok"].shape[1] else jnp.concatenate([p["tok"][xs], low], -1)
            leaf, w = walk(p, f, beam, tau, mode, None if pkey is None else jax.random.fold_in(pkey, 1), route_noise)
        h = part(p, leaf, w, xs, low, noise, None if state is None else state[i])
        last.append(h[-1])
        low = _rms(h)
        out = leaf if p["p_out"].shape[0] > 1 else 0 * leaf
        logits = logits + jnp.einsum("tbq,tbqrk,tbk->tbr", w, p["p_out"][out], h) @ p["q_out"].T
    logp = jnp.swapaxes(jax.nn.log_softmax(logits, -1), 0, 1)
    return (logp, last) if return_state else logp


# ----------------------------------------------------------------------------------------------------------- tasks

def _perm_table(perms):
    index = {q: i for i, q in enumerate(perms)}
    return np.array([[index[tuple(b[a[j]] for j in range(5))] for b in perms] for a in perms], np.int32)


def _even(q):
    q, s = list(q), 0
    for i in range(len(q)):
        while q[i] != i:
            j = q[i]
            q[i], q[j] = q[j], q[i]
            s ^= 1
    return s == 0


class Group:
    """Word problem (Merrill et al. 2024; Grazzi et al. 2025): label at t = running product x_1 ... x_t."""

    def __init__(self, table):
        self.table, self.vocab = table, len(table)
        self.classes, self.identity = len(table), 0

    def sample(self, rng, batch, length):
        x = rng.integers(0, self.vocab, (batch, length), dtype=np.int32)
        y, s = np.empty_like(x), np.full(batch, self.identity, np.int32)
        for t in range(length):
            s = self.table[s, x[:, t]]
            y[:, t] = s
        return x, y, np.ones_like(x, bool)


class Parity:
    vocab, classes = 2, 2

    def sample(self, rng, batch, length):
        x = rng.integers(0, 2, (batch, length), dtype=np.int32)
        return x, np.cumsum(x, 1) % 2, np.ones_like(x, bool)


class Cycle:
    vocab, classes = 3, 5

    def sample(self, rng, batch, length):
        x = rng.integers(0, 3, (batch, length), dtype=np.int32)
        return x, np.cumsum(x - 1, 1) % 5, np.ones_like(x, bool)


class ModArith:
    """Delétang et al. 2023: residues 0..4 alternating with + - * (tokens 5..7), * before + and -, mod 5; labelled at
    odd prefix lengths (after each residue)."""
    vocab, classes = 8, 5

    def sample(self, rng, batch, length):
        x = np.empty((batch, length), np.int32)
        x[:, ::2] = rng.integers(0, 5, (batch, (length + 1) // 2))
        x[:, 1::2] = rng.integers(5, 8, (batch, length // 2))
        y = np.zeros((batch, length), np.int32)
        acc, term = np.zeros(batch, np.int64), x[:, 0].astype(np.int64)
        y[:, 0] = term % 5
        for t in range(2, length, 2):
            op, r = x[:, t - 1], x[:, t].astype(np.int64)
            acc = np.where(op != 7, acc + term, acc)
            term = np.where(op == 5, r, np.where(op == 6, -r, term * r)) % 5
            y[:, t] = (acc + term) % 5
        valid = np.zeros((batch, length), bool)
        valid[:, ::2] = True
        return x, y, valid


_PERMS = list(itertools.permutations(range(5)))
STATE_TASKS = {  # name: (generator, train length, test length)
    "s5_word": (lambda: Group(_perm_table(_PERMS)), 32, 500),
    "a5_word": (lambda: Group(_perm_table([q for q in _PERMS if _even(q)])), 32, 500),
    "parity_check": (Parity, 40, 256),
    "cycle_navigation": (Cycle, 40, 256),
    "modular_arithmetic": (ModArith, 40, 256),
}
LM_TASKS = {"enwik8": 256, "hg38": 5}  # name: vocabulary (hg38: A C G T = 0..3, N = 4)
CLF_TASKS = {"dummy_mouse_enhancers_ensembl": 2, "demo_coding_vs_intergenomic_seqs": 2, "demo_human_or_worm": 2,
             "human_enhancers_cohn": 2, "human_enhancers_ensembl": 2, "human_ensembl_regulatory": 3,
             "human_nontata_promoters": 2, "human_ocr_ensembl": 2}  # Genomic Benchmarks: name: classes
SEQ_LENGTHS = (32, 64, 128, 256, 500)
N_EVAL, EVAL_CHUNK = 512, 128


class StateTask:
    """Train on length train_len with the loss on every labelled prefix; evaluate 512 sequences of length eval_len
    (sequence accuracy = every label up to L correct, Grazzi et al.). Streams: val (seed 20000 + s) selects the
    checkpoint, test (seed 10000 + s) is reported."""

    def __init__(self, name, seed):
        make, self.train_len, self.eval_len = STATE_TASKS[name]
        self.gen = make()
        self.vocab, self.classes = self.gen.vocab, self.gen.classes
        self.sets = {s: self.gen.sample(np.random.default_rng(o + seed), N_EVAL, self.eval_len)
                     for s, o in (("val", 20_000), ("test", 10_000))}

    def batch(self, rng, size):
        x, y, valid = self.gen.sample(rng, size, self.train_len)
        return x, y, valid.astype(np.float32)

    def carry_batch(self, rng, size, frac):
        """State passing (Buitrago Ruiz & Gu 2025; training only): (x, y, mask, prefix, use). A fraction `frac` of the
        rows (use = 1) start from the state after `prefix`: prefix and x are consecutive chunks of one stream of length
        2L, so the labels of x are the true labels of the whole stream (running product; for modular arithmetic the
        value of the whole expression, whose second chunk starts at an even position, i.e. with a residue). This is
        truncated BPTT over a stream twice the training length. Rows with use = 0 are fresh sequences from a zero
        state. (A prefix from an unrelated example would be ill-posed for modular arithmetic: the task has no
        start-of-expression token, so nothing in the input says that the carried state must be discarded.)"""
        L = self.train_len
        use = (rng.random(size) < frac).astype(np.float32)
        x, y, valid = self.gen.sample(rng, size, L)  # zero-start rows
        sx, sy, sv = self.gen.sample(rng, size, 2 * L)
        prefix = sx[:, :L]
        pick = use[:, None] > 0
        x, y, valid = np.where(pick, sx[:, L:], x), np.where(pick, sy[:, L:], y), np.where(pick, sv[:, L:], valid)
        return x, y, valid.astype(np.float32), prefix, use

    def evaluate(self, predict, params, split):
        """Sequence accuracy per length (hard path)."""
        x, y, valid = self.sets[split]
        pred = np.concatenate([np.asarray(predict(params, x[i:i + EVAL_CHUNK])) for i in range(0, len(x), EVAL_CHUNK)])
        ok = np.cumprod(np.where(valid, pred == y, True), axis=1)
        return {str(L): float(ok[:, L - 1].mean()) for L in SEQ_LENGTHS if L <= self.eval_len}


class LMTask:
    """Next-token bits per token with a zero state per window (training: --carry starts some windows from the state
    after the preceding window). enwik8: raw bytes, standard 90M / 5M / 5M split.
    hg38: the HyenaDNA / Basenji interval split (data/hg38, scripts/prepare_genomics.py), bits per nucleotide."""
    higher = False

    def __init__(self, name, cfg):
        if name == "enwik8":
            raw = np.frombuffer((ROOT / "data" / name).read_bytes(), np.uint8)
            self.train, val, test = raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]
        else:  # memory-mapped splits (train is 4.46 G tokens)
            self.train, val, test = (np.memmap(ROOT / "data" / name / f"{s}.u8", np.uint8, "r")
                                     for s in ("train", "valid", "test"))
        self.vocab = self.classes = LM_TASKS[name]
        self.T = cfg.T
        self.streams = {"val": val, "test": test}  # raw splits (streaming evaluation)
        self.sets = {"val": self._windows(val, cfg.val_windows), "test": self._windows(test, cfg.test_chars // cfg.T)}

    def _windows(self, ids, m):
        m = min(m, (len(ids) - 1) // self.T)
        ids = np.asarray(ids[:m * self.T + 1], np.int32)
        return ids[:-1].reshape(m, self.T), ids[1:].reshape(m, self.T)

    def batch(self, rng, size):
        idx = rng.integers(0, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        return self.train[idx].astype(np.int32), self.train[idx + 1].astype(np.int32), np.ones((size, self.T), np.float32)

    def carry_batch(self, rng, size, frac):
        """State passing (training only): (x, y, mask, prefix, use). Every row is a window train[i:i + T] at a random
        i >= T; prefix = the preceding window train[i - T:i]. On a fraction `frac` of the rows (use = 1) the window
        starts from the (stopped) state after its prefix: truncated BPTT over the stream. Rows with use = 0 start from
        zero. Labels are the next tokens."""
        use = (rng.random(size) < frac).astype(np.float32)
        idx = rng.integers(self.T, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        x, y, prefix = (self.train[idx + k].astype(np.int32) for k in (0, 1, -self.T))
        return x, y, np.ones((size, self.T), np.float32), prefix, use

    def evaluate(self, nll_fn, params, split):
        x, y = self.sets[split]
        nll = sum(float(nll_fn(params, x[i:i + 64], y[i:i + 64])) for i in range(0, len(x), 64))
        return nll / y.size / LN2

    def evaluate_stream(self, params, split, chunk=16384, limit=0):
        """Bits per token over the WHOLE split as one stream: zero state before the first byte, every later token
        conditioned on all earlier ones (hard routing). Batch 1, chunks of `chunk` tokens, the per-layer states carried
        from one chunk to the next. limit > 0: only the first `limit` predicted tokens of the split."""
        ids = self.streams[split]
        if limit > 0:
            ids = ids[:limit + 1]
        n = len(ids) - 1
        params = fold_tree(params)
        st = [jnp.zeros((1, p["c"].shape[-1]), jnp.float32) for p in [params["root"]] + list(params["branches"])]
        nll = 0.0
        for i in range(0, n, chunk):
            j = min(n, i + chunk)
            x = np.asarray(ids[i:j], np.int32)[None]
            y = np.asarray(ids[i + 1:j + 1], np.int32)[None]
            prev = None
            if "context" in params["root"]:  # the tokens before the chunk (zeros before the first token)
                k = params["root"]["context"].shape[0]
                prev = np.concatenate([np.zeros(k - 1, np.int32), np.asarray(ids[max(0, i - k + 1):i], np.int32)])
                prev = prev[-(k - 1):][None]
            s, st = _stream_nll(params, x, y, st, prev)
            nll += float(s)
        return nll / n / LN2, n


@jax.jit
def _stream_nll(params, x, y, state, prev=None):
    """-sum log p(y) of one chunk (1, C) from `state` (prev: the tokens before it); returns the states after it."""
    logp, last = forward(params, x, mode="hard", state=state, return_state=True, prev=prev)
    return -jnp.take_along_axis(logp, y[..., None], -1).sum(), last


class ClfTask:
    """Genomic Benchmarks (Gresova et al. 2023): label read out at the last base of each sequence (tokens A C G T N =
    0..4, pad 5 after the end; the model is causal, so padding never reaches the read-out). Official test split;
    validation = a seeded 10% of the train split (as Caduceus). Score = top-1 accuracy."""
    higher, vocab = True, 6

    def __init__(self, name, seed):
        self.classes = CLF_TASKS[name]
        d = ROOT / "data" / "genomic_benchmarks" / name
        tr, te = (np.load(d / f"{s}.npz") for s in ("train", "test"))
        perm = np.random.default_rng(seed).permutation(len(tr["y"]))
        cut = len(perm) // 10
        pick = lambda z, i: (z["x"][i].astype(np.int32), z["lengths"][i], z["y"][i].astype(np.int32))
        self.sets = {"val": pick(tr, perm[:cut]), "test": pick(te, np.arange(len(te["y"])))}
        self.train = pick(tr, perm[cut:])

    @staticmethod
    def _crop(x, lengths):  # crop to the longest sequence, rounded up to 64 (few compiled shapes)
        return x[:, :min(x.shape[1], -(-int(lengths.max()) // 64) * 64)]

    def batch(self, rng, size):
        x, lengths, y = self.train
        i = rng.integers(0, len(y), size)
        xb = self._crop(x[i], lengths[i])
        m = np.zeros(xb.shape, np.float32)
        m[np.arange(size), lengths[i] - 1] = 1.0
        return xb, np.broadcast_to(y[i][:, None], xb.shape).copy(), m

    def evaluate(self, predict, params, split):
        x, lengths, y = self.sets[split]
        order, hits = np.argsort(lengths), 0
        for j in range(0, len(order), EVAL_CHUNK):
            i = order[j:j + EVAL_CHUNK]
            pred = np.asarray(predict(params, self._crop(x[i], lengths[i])))
            hits += int((pred[np.arange(len(i)), lengths[i] - 1] == y[i]).sum())
        return hits / len(y)


# -------------------------------------------------------------------------------------------------------- training

LEAF_KEYS = ("u_in", "c", "p_out", "skew", "d_raw", "u_low")


def to_tree(p, depth):
    """Branch leaf parameters as a tree of increments (path sums: UltraFastBERT, Belcak & Wattenhofer 2023; hierarchical
    shrinkage, Agarwal et al. 2022): leaf j = (D + 1)^-1/2 times the sum of the terms of the D + 1 nodes on its
    root-to-leaf path, levels stored as arrays of 2^d rows. The top node holds the single-leaf init and every deeper
    term starts at zero, so the layer starts as one shared leaf and the leaves differ only where training splits them.
    A reparameterisation only: any leaf table is reachable, and the saved model is the summed table."""
    sc = (depth + 1) ** -0.5
    levels = {k: [p[k][:1] / sc] + [jnp.zeros((2**d,) + p[k].shape[1:], p[k].dtype) for d in range(1, depth + 1)]
              for k in LEAF_KEYS if k in p}
    return dict({k: v for k, v in p.items() if k not in levels}, levels=levels)


def leaf_table(p):
    """Tree of increments -> the per-leaf arrays (identity for a branch already folded)."""
    if "levels" not in p:
        return p
    out = {k: v for k, v in p.items() if k != "levels"}
    for k, levels in p["levels"].items():
        depth = len(levels) - 1
        out[k] = (depth + 1) ** -0.5 * sum(jnp.repeat(lv, 2**(depth - d), axis=0) for d, lv in enumerate(levels))
    return out


def fold_tree(params):
    """The summed leaf tables (the saved model has the usual arrays)."""
    return dict(params, branches=[leaf_table(p) for p in params["branches"]])


def save(path, params):
    flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in kp): np.asarray(v)
            for kp, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, **flat)


def train(cfg):
    lm = cfg.task in LM_TASKS
    task = LMTask(cfg.task, cfg) if lm else ClfTask(cfg.task, cfg.seed) if cfg.task in CLF_TASKS else \
        StateTask(cfg.task, cfg.seed)
    params = init(jax.random.PRNGKey(cfg.seed), task.vocab, task.classes, cfg.n, cfg.block, cfg.depth, cfg.layers,
                  cfg.feat, cfg.rank, cfg.out_rank, cfg.lower_rank, cfg.shared_out, cfg.branch, cfg.context_len,
                  cfg.context_bits)
    n_params = int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))
    params["branches"] = [to_tree(p, cfg.depth) for p in params["branches"]]
    warmup = min(1000, cfg.steps // 10) if cfg.warmup_frac is None else int(round(cfg.warmup_frac * cfg.steps))
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warmup, cfg.steps, cfg.lr * 0.01)
    opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)), 1000)
    opt_state = opt.init(params)

    def loss_fn(p, x, y, m, key, prefix=None, use=None):
        state = None
        if prefix is not None:  # --carry: start from the (stopped) state after the prefix, on rows with use = 1
            last = forward(p, prefix, cfg.beam, ROUTE_TAU, "st", jax.random.fold_in(key, 7), cfg.route_noise,
                           cfg.state_noise, return_state=True)[1]
            state = [jax.lax.stop_gradient(h) * use[:, None] for h in last]
        logp = forward(p, x, cfg.beam, ROUTE_TAU, "st", key, cfg.route_noise, cfg.state_noise, state=state)
        return -(jnp.take_along_axis(logp, y[..., None], -1)[..., 0] * m / m.sum()).sum()

    @jax.jit
    def step(p, s, x, y, m, key, prefix=None, use=None):
        ce, g = jax.value_and_grad(loss_fn)(p, x, y, m, key, prefix, use)
        u, s = opt.update(g, s, p)
        if cfg.branch_lr != 1.0:  # branch updates scaled after Adam
            u = dict(u, branches=jax.tree_util.tree_map(lambda v: v * cfg.branch_lr, u["branches"]))
        return optax.apply_updates(p, u), s, ce

    predict = jax.jit(lambda p, x: forward(p, x, mode="hard").argmax(-1))
    nll = jax.jit(lambda p, x, y: -jnp.take_along_axis(forward(p, x, mode="hard"), y[..., None], -1).sum())

    name = f"{cfg.task}_D{cfg.depth}_n{cfg.n}_b{cfg.block}_K{cfg.beam}" + ("" if cfg.branch == "dyn" else f"_{cfg.branch}") \
        + ("" if cfg.context_len == 1 else f"_k{cfg.context_len}") + f"{cfg.tag}_s{cfg.seed}"
    out = Path(cfg.out) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = None
    if cfg.wandb:
        import wandb
        wb = wandb.init(name=os.environ.get("WANDB_NAME", name), config=dict(vars(cfg), n_params=n_params))
    print(name, n_params, jax.devices()[0], flush=True)
    rng, log, t0 = np.random.default_rng(cfg.seed), [], time.time()
    best = {"val": float("inf") if lm else -1.0}
    for it in range(1, cfg.steps + 1):
        prefix = use = None
        if cfg.carry > 0:
            assert isinstance(task, (StateTask, LMTask)), "--carry: state and LM tasks"
            x, y, m, prefix, use = task.carry_batch(rng, cfg.batch, cfg.carry)
        else:
            x, y, m = task.batch(rng, cfg.batch)
        params, opt_state, ce = step(params, opt_state, x, y, m, jax.random.PRNGKey(cfg.seed * 1_000_003 + it),
                                     prefix, use)
        if it % cfg.eval_every and it != cfg.steps:
            continue
        rec = {"step": it, "loss": float(ce), "time_s": round(time.time() - t0, 1)}
        if lm:
            rec["val_bpc"] = val = task.evaluate(nll, params, "val")
            better = val < best["val"]
            if cfg.stream_val > 0:  # the first stream_val tokens of the validation split as one stream (not selection)
                rec["val_stream"] = task.evaluate_stream(params, "val", cfg.stream_chunk, cfg.stream_val)[0]
        elif cfg.task in CLF_TASKS:
            rec["val_acc"] = val = task.evaluate(predict, params, "val")
            better = val > best["val"]
        else:
            rec["val_seq"] = task.evaluate(predict, params, "val")
            val = rec["val_seq"][str(task.eval_len)]
            better = val > best["val"]
        if better:
            best = {"val": val, "step": it, "params": jax.device_get(params)}
            if "val_stream" in rec:
                best["val_stream"] = rec["val_stream"]
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if wb is not None:
            wb.log({"loss": rec["loss"], "val": val, "best_val": best["val"],
                    **({"val_stream": rec["val_stream"]} if "val_stream" in rec else {})}, step=it)
    res = {"config": vars(cfg), "name": name, "n_params": n_params, "log": log, "best_step": best.get("step"),
           "best_val": best["val"], "train_s": round(time.time() - t0, 1),
           "code_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()}
    if "val_stream" in best:  # validation stream at the best checkpoint
        res["val_stream"] = best["val_stream"]
    if "params" in best:  # test at the best-validation checkpoint (the reported number)
        res["test"] = task.evaluate(nll if lm else predict, best["params"], "test")
        save(out.with_suffix(".npz"), fold_tree(best["params"]))
        if lm and cfg.stream_test:  # the final protocol: the full test split as one stream
            res["test_stream"], res["test_stream_tokens"] = task.evaluate_stream(best["params"], "test",
                                                                                  cfg.stream_chunk)
    out.with_suffix(".json").write_text(json.dumps(res))
    if wb is not None:
        wb.summary.update({"best_val": best["val"], "test": res.get("test")})
        if "test_stream" in res:
            wb.summary.update({"test_stream": res["test_stream"]})
        if "val_stream" in res:
            wb.summary.update({"val_stream": res["val_stream"]})
        wb.finish()
    print("DONE", name, "best val", best["val"], "test", res.get("test"),
          *(("test_stream", res["test_stream"]) if "test_stream" in res else ()), flush=True)
    return best["val"]


def trial(cfg, argv):
    """One W&B sweep trial: one process per (task, seed) on this GPU. Score of a task = mean over seeds of the best
    validation value (state: sequence accuracy at the test length; LM: bits per byte, negated so higher is better);
    objective = the worst task's score (--tasks s5_word,modular_arithmetic: one setting must pass both)."""
    import wandb
    wb = wandb.init()
    seeds = cfg.seeds.split(",")
    names = cfg.tasks.split(",") if cfg.tasks else [cfg.task]
    rest, skip = [], False  # argv without the command, --seeds, --tasks and --task
    for a in argv[1:]:
        if skip or a.split("=")[0] in ("--seeds", "--tasks", "--task"):
            skip = a in ("--seeds", "--tasks", "--task")
            continue
        rest.append(a)
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_") or k in
           ("WANDB_API_KEY", "WANDB_SILENT", "WANDB_MODE", "WANDB_ENTITY", "WANDB_DIR", "WANDB_BASE_URL")}
    env.update(WANDB_PROJECT=wb.project, WANDB_RUN_GROUP=f"trial_{wb.id}")
    out = Path(cfg.out) / wb.id
    out.mkdir(parents=True, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, __file__, "train", *rest, "--task", t, "--seed", s, "--out", str(out)],
                              env=dict(env, WANDB_NAME=f"trial_{wb.id}_{t}_s{s}"),
                              stdout=open(out / f"{t}_s{s}.log", "w"), stderr=subprocess.STDOUT)
             for t in names for s in seeds]
    for q in procs:
        q.wait()
    scores, summary = [], {}
    for t in names:
        vals = []
        for s in seeds:
            hits = list(out.glob(f"{t}_*_s{s}.json"))
            vals.append(json.loads(hits[0].read_text())["best_val"] if hits else None)
            summary[f"{t}/s{s}"] = vals[-1]
        ok = [v for v in vals if v is not None]
        lm = t in LM_TASKS
        score = (-sum(ok) / len(ok) if lm else sum(ok) / len(seeds)) if ok else (-99.0 if lm else 0.0)
        summary[f"{t}/score"] = score
        scores.append(score)
    wb.summary.update({"objective": min(scores), **summary})
    wb.finish()


def sweep(path):
    """Create one W&B sweep (method: bayes default, or grid) per entry of `variants` in a YAML config (keys: project,
    method, run_cap, fixed, search,
    variants: {name: overrides}); prints the sweep paths. Search values: list, or {log_uniform: [lo, hi]}."""
    import wandb
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    param = lambda v: ({"values": v} if isinstance(v, list) else {"distribution": "log_uniform_values",
                       "min": v["log_uniform"][0], "max": v["log_uniform"][1]} if isinstance(v, dict) else {"value": v})
    paths = []
    for name, over in cfg.get("variants", {"main": {}}).items():
        values = {**cfg.get("fixed", {}), **cfg.get("search", {}), **(over or {})}
        spec = {"name": f"{Path(path).stem}-{name}", "method": cfg.get("method", "bayes"), "program": str(Path(__file__).resolve()),
                "metric": {"name": "objective", "goal": "maximize"}, "parameters": {k: param(v) for k, v in values.items()},
                "command": ["${env}", "${interpreter}", "${program}", "train", "${args}"]}
        if "run_cap" in cfg:
            spec["run_cap"] = cfg["run_cap"]
        sid = wandb.sweep(spec, project=cfg["project"])
        paths.append(f"{wandb.Api().default_entity}/{cfg['project']}/{sid}")
    print(" ".join(paths))


def parser():
    ap = argparse.ArgumentParser(description="Yggdrasil")
    ap.add_argument("command", choices=["train", "sweep"])
    ap.add_argument("config", nargs="?", help="sweep: YAML config")
    add = ap.add_argument
    add("--task", choices=[*STATE_TASKS, *LM_TASKS, *CLF_TASKS])
    add("--tasks", default="", help="W&B trial over several tasks (comma list); objective = the worst task's score")
    add("--seed", type=int, default=0)
    add("--seeds", default="", help="comma list: run one process per seed and report the mean (W&B sweep trial)")
    # model
    add("--layers", type=int, default=2)
    add("--n", type=int, default=128)
    add("--block", type=int, default=16)
    add("--depth", type=int, default=8)
    add("--feat", type=int, default=16)
    add("--rank", type=int, default=16, help="rank of the token input map U_r v(x)")
    add("--out_rank", type=int, default=8)
    add("--lower_rank", type=int, default=16)
    add("--shared_out", type=_flag, default=False)
    add("--branch", default="dyn", choices=["dyn", "out", "tok"])
    add("--context_len", type=int, default=1, help="root leaf = the last k tokens (1: the token)")
    add("--context_bits", type=int, default=16, help="context root: at most 2^bits leaves (hashed when "
        "vocab^k is larger)")
    # training
    add("--steps", type=int, default=40_000)
    add("--batch", type=int, default=256)
    add("--T", type=int, default=256, help="LM window length")
    add("--lr", type=float, default=1e-3)
    add("--branch_lr", type=float, default=1.0)
    add("--beam", type=int, default=8)
    add("--route_noise", type=float, default=0.05, help="logistic noise on the node tests (training)")
    add("--state_noise", type=float, default=0.0, help="Gaussian noise on each layer's input drive (training)")
    add("--carry", type=float, default=0.0, help="state and LM tasks: fraction of training sequences that start "
        "from a carried state (state passing; LM: the state after the preceding window of the stream; training only)")
    add("--warmup_frac", type=float, default=None,
        help="LR warmup as a fraction of --steps (default: min(1000, steps // 10), i.e. 10%% at 10k steps)")
    # evaluation / output
    add("--eval_every", type=int, default=4_000)
    add("--val_windows", type=int, default=512)
    add("--test_chars", type=int, default=5_000_000)
    add("--stream_test", type=_flag, default=False,
        help="LM: also score the full test split as one stream (state carried from a zero start): json 'test_stream'")
    add("--stream_chunk", type=int, default=16384, help="tokens per chunk of the streaming test (batch 1)")
    add("--stream_val", type=int, default=0, help="LM, > 0: at each evaluation also score the first N tokens of the "
        "validation split as one stream (json 'val_stream' at the best checkpoint; selection stays on val_bpc)")
    add("--out", default=str(ROOT / "results"))
    add("--tag", default="")
    add("--wandb", type=_flag, default=False)
    return ap


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "sweep":
        sweep(args.config)
    elif args.seeds:
        trial(args, sys.argv[1:])
    else:
        assert args.task, "--task (or --tasks with --seeds)"
        train(args)
