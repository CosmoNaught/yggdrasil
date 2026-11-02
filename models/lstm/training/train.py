"""LSTM baseline: token embedding, torch.nn.LSTM, linear head."""
# Upstream: torch.nn.LSTM, PyTorch (https://github.com/pytorch/pytorch), BSD-3-Clause.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def add_args(ap):
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--forget_bias", type=float, default=0.0)


class LSTM(nn.Module):
    def __init__(self, vocab, classes, d, layers, forget_bias=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True)
        self.head = nn.Linear(d, classes)
        if forget_bias:
            # Gate order is (i, f, g, o); split the forget bias over b_ih and b_hh
            with torch.no_grad():
                for name, b in self.rnn.named_parameters():
                    if name.startswith("bias"):
                        b[d:2 * d] = forget_bias / 2

    def forward(self, tokens):
        return self.head(self.rnn(self.emb(tokens))[0])


def build(cfg, vocab, classes):
    return LSTM(vocab, classes, cfg.d, cfg.layers, getattr(cfg, "forget_bias", 0.0))


if __name__ == "__main__":
    import trainer
    trainer.main("lstm", build, add_args)
