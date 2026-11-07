"""State-tracking tasks: train on lengths 3 to 40, validate and test on lengths 40 to 256, score the final answer."""
import itertools

import numpy as np

TRAIN_LENS = (3, 40)
TEST_LENS = (40, 256)
N_VAL, N_TEST, N_CURVE = 4096, 8192, 1024
SPLIT_SEED = {"val": 20_000, "test": 10_000, "curve": 30_000}  # + run seed; disjoint from the training stream
EVAL_CHUNK = 128
SOLVED = 0.90


def _perm_table(perms):
    """Multiplication table of a permutation group, (n, n) indices into perms."""
    index = {q: i for i, q in enumerate(perms)}
    return np.array([[index[tuple(b[a[j]] for j in range(len(a)))] for b in perms] for a in perms], np.int32)


def _even(q):
    """Whether permutation q is even (count the swaps that sort it)."""
    q, s = list(q), 0
    for i in range(len(q)):
        while q[i] != i:
            j = q[i]
            q[i], q[j] = q[j], q[i]
            s ^= 1
    return s == 0


class Group:
    """Word problem of a group: the label at t is the product x_1 ... x_t."""

    def __init__(self, table):
        self.table, self.vocab = table, len(table)
        self.classes, self.identity = len(table), 0

    def sample(self, rng, batch, length):
        """(x, y, valid), (batch, length) each; valid marks the labelled positions (every one here)."""
        x = rng.integers(0, self.vocab, (batch, length), dtype=np.int32)
        y, s = np.empty_like(x), np.full(batch, self.identity, np.int32)
        for t in range(length):
            s = self.table[s, x[:, t]]
            y[:, t] = s
        return x, y, np.ones_like(x, bool)


class Parity:
    """Bits; the label at t is the parity of x_1 ... x_t."""
    vocab, classes = 2, 2

    def sample(self, rng, batch, length):
        x = rng.integers(0, 2, (batch, length), dtype=np.int32)
        return x, np.cumsum(x, 1) % 2, np.ones_like(x, bool)


CYCLE = 5  # positions on the cycle
MOVES = 3  # token m moves by m - 1: back, stay, forward


class Cycle:
    """Moves on a cycle; the label at t is the position after x_1 ... x_t."""
    vocab, classes = MOVES, CYCLE

    def sample(self, rng, batch, length):
        x = rng.integers(0, MOVES, (batch, length), dtype=np.int32)
        return x, np.cumsum(x - 1, 1) % CYCLE, np.ones_like(x, bool)


MOD = 5
ADD, SUB, MUL = 5, 6, 7  # operator tokens after the residues 0..4


class ModArith:
    """Residues alternating with + - *, evaluated mod 5 with precedence; labels after each residue."""
    vocab, classes = MUL + 1, MOD

    def sample(self, rng, batch, length):
        x = np.empty((batch, length), np.int32)
        x[:, ::2] = rng.integers(0, MOD, (batch, (length + 1) // 2))
        x[:, 1::2] = rng.integers(ADD, MUL + 1, (batch, length // 2))
        y = np.zeros((batch, length), np.int32)
        acc, term = np.zeros(batch, np.int64), x[:, 0].astype(np.int64)
        y[:, 0] = term % MOD
        for t in range(2, length, 2):
            op, r = x[:, t - 1], x[:, t].astype(np.int64)
            acc = np.where(op != MUL, acc + term, acc)
            term = np.where(op == ADD, r, np.where(op == SUB, -r, term * r)) % MOD
            y[:, t] = (acc + term) % MOD
        valid = np.zeros((batch, length), bool)
        valid[:, ::2] = True
        return x, y, valid


N_ITEMS = 5  # S5 and A5 permute 5 items
_PERMS = list(itertools.permutations(range(N_ITEMS)))
GENERATORS = {
    "s5_word": lambda: Group(_perm_table(_PERMS)),
    "a5_word": lambda: Group(_perm_table([q for q in _PERMS if _even(q)])),
    "parity_check": Parity,
    "cycle_navigation": Cycle,
    "modular_arithmetic": ModArith,
}


def last_label(valid, lengths):
    """Index of the last labelled position among the first lengths[i] tokens of each row; (B, L), (B,) -> (B,)."""
    pos = np.arange(valid.shape[1])[None]
    return np.where(valid & (pos < lengths[:, None]), pos, -1).max(1)


class StateTask:
    """Fresh training batches of random lengths, fixed validation and test sets drawn from the run seed."""
    kind = "state"

    def __init__(self, name, seed):
        self.name, self.gen = name, GENERATORS[name]()
        self.vocab, self.classes = self.gen.vocab, self.gen.classes
        self.chance = 1.0 / self.classes
        self.train_len, self.eval_len = TRAIN_LENS[1], TEST_LENS[1]
        self.sets = {}
        for split, n in (("val", N_VAL), ("test", N_TEST)):
            rng = np.random.default_rng(SPLIT_SEED[split] + seed)
            # rows are right-padded to TEST_LENS[1]; every model is causal, so padding cannot change the final answer
            x, y, valid = self.gen.sample(rng, n, TEST_LENS[1])
            lengths = rng.integers(TEST_LENS[0], TEST_LENS[1] + 1, n)
            self.sets[split] = (x, y, last_label(valid, lengths), lengths)
        self.curve_set = self.gen.sample(np.random.default_rng(SPLIT_SEED["curve"] + seed), N_CURVE, TEST_LENS[1])

    def batch(self, rng, size):
        """(x, y, mask), (size, 40); mask is 1 on labelled positions within each sequence's length."""
        x, y, valid = self.gen.sample(rng, size, TRAIN_LENS[1])
        lengths = rng.integers(TRAIN_LENS[0], TRAIN_LENS[1] + 1, size)
        m = valid & (np.arange(TRAIN_LENS[1])[None] < lengths[:, None])
        return x, y, m.astype(np.float32)

    def scaled(self, acc):
        """Accuracy scaled to 0 at chance and 1 at perfect."""
        return (acc - self.chance) / (1.0 - self.chance)

    @staticmethod
    def _predict(predict, x, chunk):
        return np.concatenate([np.asarray(predict(x[i:i + chunk])) for i in range(0, len(x), chunk)])

    def evaluate(self, predict, split, chunk=EVAL_CHUNK):
        """predict (b, 256) tokens -> (b, 256) labels; returns {"acc", "scaled", "chance", "n"} of the final answer."""
        x, y, last, _ = self.sets[split]
        pred = self._predict(predict, x, chunk)
        rows = np.arange(len(x))
        acc = float((pred[rows, last] == y[rows, last]).mean())
        return {"scaled": self.scaled(acc), "acc": acc, "chance": self.chance, "n": len(x)}

    def curve(self, predict, chunk=EVAL_CHUNK):
        """Final-answer and sequence accuracy at every length 1..256 on N_CURVE sequences."""
        x, y, valid = self.curve_set
        pred = self._predict(predict, x, chunk)
        hit = pred == y
        Ls = np.arange(1, x.shape[1] + 1)
        # a length whose last token is unlabelled (modular arithmetic, even L) repeats the answer of L - 1
        final = [float(hit[np.arange(len(x)), last_label(valid, np.full(len(x), L))].mean()) for L in Ls]
        seq = np.cumprod(np.where(valid, hit, True), axis=1).mean(0)
        return {"length": Ls.tolist(), "final_acc": final, "final_scaled": [self.scaled(a) for a in final],
                "seq_acc": [float(v) for v in seq], "chance": self.chance}
