"""Sizes every baseline takes from its task, and the wrapper of the library language models."""
from torch import nn

import tasks


def dims(task):
    """(input vocabulary, output classes, longest sequence) of a task."""
    if task in tasks.TEXT:
        from tasks.text import WINDOW
        return tasks.TEXT[task], tasks.TEXT[task], WINDOW[task]
    from tasks.state import GENERATORS, TEST_LENS
    g = GENERATORS[task]()
    return g.vocab, g.classes, TEST_LENS[1]


def shape(config, defaults):
    """The model's shape keys of config, with defaults for the missing ones."""
    return {k: config.get(k, v) for k, v in defaults.items()}


def n_heads(s):
    """s["n_heads"], or d_model / head_dim when it is None."""
    if s["n_heads"] is not None:
        return s["n_heads"]
    if s["d_model"] % s["head_dim"]:
        raise ValueError(f"d_model = {s['d_model']} is not divisible by head_dim = {s['head_dim']} (set n_heads)")
    return s["d_model"] // s["head_dim"]


class LibraryLM(nn.Module):
    """A library causal LM over max(vocab, classes) tokens; int64 (B, L) -> logits (B, L, classes)."""

    def __init__(self, model, classes, **forward_kwargs):
        super().__init__()
        self.model, self.classes, self.forward_kwargs = model, classes, forward_kwargs

    def forward(self, tokens):
        return self.model(input_ids=tokens, **self.forward_kwargs).logits[..., :self.classes]
