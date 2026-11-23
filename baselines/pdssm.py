"""PD-SSM model from its authors' code (third_party/pdssm), trained from scratch; state tasks only."""
from .common import dims, shape

DEFAULTS = {"d_model": 128, "n_layers": 2}
AMP = None  # trained in fp32
DICTIONARY_SIZE = 8  # the authors' defaults
DROPOUT = 0.01


def build(config):
    from .third_party.pdssm.pdssm import PD
    vocab, classes, _ = dims(config["task"])
    s = shape(config, DEFAULTS)
    # state size = embedding size, as the authors' train.py builds it; logits at every position. The authors' code
    # keeps its states on the GPU whenever one is present, so the model runs on the GPU there.
    return PD(output_size=classes, input_size=vocab, state_size=s["d_model"], embed_size=s["d_model"],
              dictionary_size=DICTIONARY_SIZE, return_all_outputs=True, num_layers=s["n_layers"],
              dropout_rate=DROPOUT)
