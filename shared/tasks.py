"""Tasks and evaluation of models/yggdrasil/training/train.py, numpy only (models/yggdrasil/training/train.py imports JAX at the top, so it cannot be imported
from the PyTorch venv). The block between the COPY markers is models/yggdrasil/training/train.py lines 201-283 verbatim (generators, task table,
SEQ_LENGTHS, N_EVAL, EVAL_CHUNK); StateTask and LMTask repeat models/yggdrasil/training/train.py's classes with the model call left to the
caller. check_protocol.py (run with the JAX venv) asserts that every train batch, val set and test set equals models/yggdrasil/training/train.py's.
"""
import itertools
from pathlib import Path

import numpy as np

YGG = Path(__file__).resolve().parents[1]
LN2 = float(np.log(2.0))

# ---- COPY of models/yggdrasil/training/train.py lines 201-283 (do not edit) ----
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
# ---- end COPY ----


class StateTask:
    """models/yggdrasil/training/train.py StateTask: train on length train_len with the loss on every labelled prefix; 512 val (seed 20000 + s)
    and 512 test (seed 10000 + s) sequences of length eval_len; sequence accuracy = every label up to L correct."""

    def __init__(self, name, seed):
        make, self.train_len, self.eval_len = STATE_TASKS[name]
        self.gen = make()
        self.vocab, self.classes = self.gen.vocab, self.gen.classes
        self.sets = {s: self.gen.sample(np.random.default_rng(o + seed), N_EVAL, self.eval_len)
                     for s, o in (("val", 20_000), ("test", 10_000))}

    def batch(self, rng, size):
        x, y, valid = self.gen.sample(rng, size, self.train_len)
        return x, y, valid.astype(np.float32)

    def evaluate(self, predict, split, chunk=EVAL_CHUNK):
        """predict(x int (b, L)) -> argmax labels (b, L) numpy. Returns {str(L): sequence accuracy}."""
        x, y, valid = self.sets[split]
        pred = np.concatenate([np.asarray(predict(x[i:i + chunk])) for i in range(0, len(x), chunk)])
        ok = np.cumprod(np.where(valid, pred == y, True), axis=1)
        return {str(L): float(ok[:, L - 1].mean()) for L in SEQ_LENGTHS if L <= self.eval_len}


class LMTask:
    """models/yggdrasil/training/train.py LMTask (enwik8): raw bytes, 90M / 5M / 5M split, windows of T from a zero state, bits per byte."""

    def __init__(self, name, T, val_windows, test_chars):
        assert name == "enwik8"
        raw = np.frombuffer((YGG / "data" / name).read_bytes(), np.uint8)
        self.train, val, test = raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]
        self.vocab = self.classes = LM_TASKS[name]
        self.T = T
        self.sets = {"val": self._windows(val, val_windows), "test": self._windows(test, test_chars // T)}

    def _windows(self, ids, m):
        m = min(m, (len(ids) - 1) // self.T)
        ids = np.asarray(ids[:m * self.T + 1], np.int32)
        return ids[:-1].reshape(m, self.T), ids[1:].reshape(m, self.T)

    def batch(self, rng, size):
        idx = rng.integers(0, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        return self.train[idx].astype(np.int32), self.train[idx + 1].astype(np.int32), np.ones((size, self.T), np.float32)

    def evaluate(self, nll_fn, split, chunk=64):
        """nll_fn(x, y) -> summed negative log-likelihood (nats). Returns bits per byte."""
        x, y = self.sets[split]
        nll = sum(float(nll_fn(x[i:i + chunk], y[i:i + chunk])) for i in range(0, len(x), chunk))
        return nll / y.size / LN2
