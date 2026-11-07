"""Gated DeltaProduct language model of fla, with negative eigenvalues, and a forget gate on text and DNA."""
import torch

import tasks

from .common import LibraryLM, dims, n_heads, shape

DEFAULTS = {"d_model": 128, "n_layers": 2, "head_dim": 32, "n_heads": None, "d_ff": None}  # None: fla defaults
AMP = torch.bfloat16  # the chunked kernel does not accept fp32 inputs
HOUSEHOLDER = {"deltaproduct2": 2, "deltaproduct4": 4}  # Householder factors per step, set by the model name


def build(config):
    from fla.models import GatedDeltaProductConfig, GatedDeltaProductForCausalLM
    vocab, classes, _ = dims(config["task"])
    s = shape(config, DEFAULTS)
    cfg = GatedDeltaProductConfig(hidden_size=s["d_model"], num_hidden_layers=s["n_layers"], head_dim=s["head_dim"],
                                  num_heads=n_heads(s), num_householder=HOUSEHOLDER[config["model"]],
                                  intermediate_size=s["d_ff"], use_forget_gate=config["task"] in tasks.TEXT,
                                  allow_neg_eigval=True, vocab_size=max(vocab, classes), tie_word_embeddings=False,
                                  fuse_norm=False, fuse_cross_entropy=False, use_cache=False)
    return LibraryLM(GatedDeltaProductForCausalLM(cfg), classes, use_cache=False)
