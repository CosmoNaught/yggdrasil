"""GRU baseline: token embedding, torch.nn.GRU, linear head."""
# Upstream: torch.nn.GRU, PyTorch (https://github.com/pytorch/pytorch), BSD-3-Clause.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch.nn as nn  # noqa: E402


def add_args(ap):
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)


class RNN(nn.Module):
    def __init__(self, cell, vocab, classes, d, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = cell(d, d, num_layers=layers, batch_first=True)
        self.head = nn.Linear(d, classes)

    def forward(self, tokens):
        return self.head(self.rnn(self.emb(tokens))[0])


def build(cfg, vocab, classes):
    return RNN(nn.GRU, vocab, classes, cfg.d, cfg.layers)


if __name__ == "__main__":
    import trainer
    trainer.main("gru", build, add_args)
