"""GRU: token embedding, torch.nn.GRU, linear head."""
from torch import nn

from .common import dims, shape

DEFAULTS = {"d_model": 128, "n_layers": 2}
AMP = None  # trained in fp32


class RNNLM(nn.Module):
    """Embedding, a stacked torch RNN cell, linear head; int64 (B, L) -> logits (B, L, classes)."""

    def __init__(self, cell, vocab, classes, d_model, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.rnn = cell(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, classes)

    def forward(self, tokens):
        return self.head(self.rnn(self.emb(tokens))[0])


def build(config):
    vocab, classes, _ = dims(config["task"])
    return RNNLM(nn.GRU, vocab, classes, **shape(config, DEFAULTS))
