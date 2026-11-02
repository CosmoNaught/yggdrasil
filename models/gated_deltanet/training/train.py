"""Gated DeltaNet LM using fla.models.GatedDeltaNetForCausalLM (flash-linear-attention 0.5.2, MIT)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch.nn as nn  # noqa: E402


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


def add_args(ap):
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=None, help="number of heads (default d / head_dim)")
    ap.add_argument("--ff", type=int, default=None, help="MLP width (fla intermediate_size; default: library rule)")
    ap.add_argument("--expand_v", type=float, default=2.0, help="value expansion (library default 2)")
    ap.add_argument("--allow_neg_eigval", type=_flag, default=None,
                    help="default: true on the state tasks, false on enwik8 / hg38")


class GatedDeltaNetLM(nn.Module):
    def __init__(self, vocab, classes, d, layers, head_dim, expand_v=2.0, neg=True, heads=None, ff=None):
        super().__init__()
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        self.classes = classes
        cfg = GatedDeltaNetConfig(hidden_size=d, num_hidden_layers=layers, head_dim=head_dim,
                                  num_heads=heads or d // head_dim, intermediate_size=ff, expand_v=expand_v,
                                  use_gate=True, use_short_conv=True, conv_size=4,
                                  allow_neg_eigval=neg, hidden_ratio=4, norm_eps=1e-6, vocab_size=max(vocab, classes),
                                  tie_word_embeddings=False, fuse_norm=False, fuse_swiglu=False,
                                  fuse_cross_entropy=False, use_cache=False)
        self.model = GatedDeltaNetForCausalLM(cfg)

    def forward(self, tokens):
        return self.model(input_ids=tokens, use_cache=False).logits[..., :self.classes]


def build(cfg, vocab, classes):
    if getattr(cfg, "allow_neg_eigval", None) is None:  # [-1, 1] eigenvalues on state tasks
        cfg.allow_neg_eigval = cfg.task not in ("enwik8", "hg38")
    heads = getattr(cfg, "heads", None)
    if heads is None and cfg.d % cfg.head_dim:
        raise ValueError(f"d = {cfg.d} not divisible by head_dim {cfg.head_dim} (pass --heads)")
    return GatedDeltaNetLM(vocab, classes, cfg.d, cfg.layers, cfg.head_dim, getattr(cfg, "expand_v", 2.0),
                           cfg.allow_neg_eigval, heads, getattr(cfg, "ff", None))


if __name__ == "__main__":
    import trainer
    trainer.main("gated_deltanet", build, add_args)
