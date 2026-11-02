"""LSTM training.

Source: PyTorch torch.nn.LSTM (cuDNN kernels on GPU), torch 2.10.0, https://github.com/pytorch/pytorch (BSD-3-Clause).
Model: token embedding (vocab x d), torch.nn.LSTM (d -> d, --layers layers, batch_first), linear head with bias.
--forget_bias sets the forget-gate bias at initialisation (split equally over b_ih and b_hh; default 0 = PyTorch init).

    python models/lstm/training/train.py --task <task> --d <width> --layers <layers> [--forget_bias <b>] --lr <lr> --seed <seed>

"""
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
        if forget_bias:  # gate order i, f, g, o: the forget gate is rows d..2d of both biases
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
