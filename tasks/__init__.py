"""Datasets and scoring shared by every model.

    task = tasks.get("enwik8")                  # or "hg38", "s5_word", "a5_word", "parity_check", ...
    x, y, mask = task.batch(rng, 64)            # int32 (B, L), int32 (B, L), float32 (B, L)
    task.evaluate(nll, "val")                   # text: nll(x, y) -> summed nats; returns bits per token
    task.evaluate(predict, "test")              # state: predict(x) -> labels; returns {"acc", "scaled", ...}
    tasks.solved(acc)                           # final-answer accuracy >= 0.90
"""
from .state import GENERATORS as _GENERATORS
from .state import SOLVED, StateTask
from .text import VOCAB as TEXT
from .text import TextTask

STATE = tuple(_GENERATORS)
NAMES = (*TEXT, *STATE)


def get(name, seed=0, seq_len=None, val_windows=512, test_tokens=None):
    """Task by name. seed draws the state tasks' evaluation sets; seq_len, val_windows, test_tokens apply to text."""
    if name in TEXT:
        return TextTask(name, seq_len, val_windows, test_tokens)
    if name in STATE:
        return StateTask(name, seed)
    raise KeyError(f"unknown task {name}; expected one of {NAMES}")


def solved(acc):
    """The criterion of Deletang et al. (2023) on final-answer accuracy."""
    return acc >= SOLVED
