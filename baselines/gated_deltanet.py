"""Gated DeltaNet language model of fla, with eigenvalues in [-1, 1] on the state tasks."""
import tasks

from .common import LibraryLM, dims, n_heads, shape

DEFAULTS = {"d_model": 128, "n_layers": 2, "head_dim": 32, "n_heads": None, "d_ff": None}  # None: fla defaults
AMP = None  # trained in fp32


def build(config):
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    vocab, classes, _ = dims(config["task"])
    s = shape(config, DEFAULTS)
    cfg = GatedDeltaNetConfig(hidden_size=s["d_model"], num_hidden_layers=s["n_layers"], head_dim=s["head_dim"],
                              num_heads=n_heads(s), intermediate_size=s["d_ff"],
                              allow_neg_eigval=config["task"] in tasks.STATE, vocab_size=max(vocab, classes),
                              tie_word_embeddings=False, fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
                              use_cache=False)
    return LibraryLM(GatedDeltaNetForCausalLM(cfg), classes, use_cache=False)
