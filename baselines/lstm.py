"""LSTM: token embedding, torch.nn.LSTM, linear head."""
from torch import nn

from .common import dims, shape
from .gru import RNNLM

DEFAULTS = {"d_model": 128, "n_layers": 2}
AMP = None  # trained in fp32


def build(config):
    vocab, classes, _ = dims(config["task"])
    return RNNLM(nn.LSTM, vocab, classes, **shape(config, DEFAULTS))
