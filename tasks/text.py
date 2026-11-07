"""enwik8 bytes and hg38 bases as next-token prediction in windows of seq_len tokens, scored in bits per token."""
import math
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("YGGDRASIL_DATA", Path(__file__).resolve().parents[1] / "data"))
N_TOKEN = 4  # hg38: A C G T = 0..3, anything else = N_TOKEN
VOCAB = {"enwik8": 256, "hg38": N_TOKEN + 1}
WINDOW = {"enwik8": 256, "hg38": 1024}
LN2 = math.log(2.0)
EVAL_CHUNK = 64
ENWIK8_SPLIT = (90_000_000, 95_000_000)  # train / val / test = 90M / 5M / 5M bytes
SPLIT_FILE = ("train", "valid", "test")


def streams(name, data=None):
    """(train, val, test) uint8 token arrays from data (default DATA); hg38 uses the Basenji / HyenaDNA intervals."""
    data = Path(data or DATA)
    if name == "enwik8":
        raw = np.frombuffer((data / name).read_bytes(), np.uint8)
        a, b = ENWIK8_SPLIT
        return raw[:a], raw[a:b], raw[b:]
    return tuple(np.memmap(data / name / f"{s}.u8", np.uint8, "r") for s in SPLIT_FILE)


def windows(ids, seq_len, m):
    """Up to m consecutive windows of ids -> (x, y), int32 (m, seq_len), y shifted by one."""
    m = min(m, (len(ids) - 1) // seq_len)
    ids = np.asarray(ids[:m * seq_len + 1], np.int32)
    return ids[:-1].reshape(m, seq_len), ids[1:].reshape(m, seq_len)


class TextTask:
    """Random training windows; validation on the first val_windows windows, test on test_tokens (default all)."""
    kind = "text"

    def __init__(self, name, seq_len=None, val_windows=512, test_tokens=None):
        self.name, self.seq_len = name, seq_len or WINDOW[name]
        self.vocab = self.classes = VOCAB[name]
        self.train_len = self.eval_len = self.seq_len
        self.train, val, test = streams(name)
        self.streams = {"val": val, "test": test}
        n_test = len(test) if test_tokens is None else test_tokens // self.seq_len
        self.sets = {"val": windows(val, self.seq_len, val_windows), "test": windows(test, self.seq_len, n_test)}

    def batch(self, rng, size):
        """(x, y, mask), (size, seq_len); every window starts at a uniform random offset of the training stream."""
        idx = rng.integers(0, len(self.train) - self.seq_len - 1, size)[:, None] + np.arange(self.seq_len)[None]
        x, y = self.train[idx].astype(np.int32), self.train[idx + 1].astype(np.int32)
        return x, y, np.ones((size, self.seq_len), np.float32)

    def evaluate(self, nll, split, chunk=EVAL_CHUNK):
        """nll(x, y) -> summed negative log-likelihood in nats; returns bits per token over the split's windows."""
        x, y = self.sets[split]
        total = sum(float(nll(x[i:i + chunk], y[i:i + chunk])) for i in range(0, len(x), chunk))
        return total / y.size / LN2
