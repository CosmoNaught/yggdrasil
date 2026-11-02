"""DeltaProduct training (DeltaProduct-2 and DeltaProduct-4 by --householder).

Source: flash-linear-attention 0.5.2 (pip flash-linear-attention==0.5.2, fla-core==0.5.2),
https://github.com/fla-org/flash-linear-attention (MIT), which hosts the authors' Gated DeltaProduct (Siems et al.
2025) with its Triton chunk kernels. Used unchanged: fla.models.GatedDeltaProductForCausalLM.
Settings: --householder Householder steps per token (2 or 4), --heads heads of --head_dim (default d / head_dim), negative
eigenvalues allowed on every task (DeltaProduct_n^[-1,1]; Siems et al. 2025 use [-1, 1] in their language-model
experiments too), no forget gate (--forget_gate true adds it), library defaults otherwise
(short convolutions, output gate, SwiGLU MLP ratio 4; width --ff when given, else the library's rounding to 256),
untied head, unfused norms and loss.
fla's DeltaProduct chunk kernel rejects fp32 inputs, so training and evaluation run under bf16 autocast (fp32 weights
and Adam state, logits and loss in fp32).

    python models/deltaproduct/training/train.py --task <task> --householder <2|4> --d <width> --layers <layers> --head_dim <head_dim> --lr <lr> --seed <seed>

"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

AMP = torch.bfloat16


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


def add_args(ap):
    ap.add_argument("--householder", type=int, default=2, choices=(2, 4))
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=None, help="number of heads (default d / head_dim); fla takes "
                    "num_heads and head_dim independently, so heads x head_dim need not equal d")
    ap.add_argument("--ff", type=int, default=None, help="MLP width (fla intermediate_size; default: library rule)")
    ap.add_argument("--forget_gate", type=_flag, default=False)


class DeltaProductLM(nn.Module):
    def __init__(self, vocab, classes, d, layers, head_dim, householder, forget_gate=False, heads=None, ff=None):
        super().__init__()
        from fla.models import GatedDeltaProductConfig, GatedDeltaProductForCausalLM
        self.classes = classes
        cfg = GatedDeltaProductConfig(hidden_size=d, num_hidden_layers=layers, head_dim=head_dim,
                                      num_heads=heads or d // head_dim, num_householder=householder,
                                      intermediate_size=ff,
                                      use_forget_gate=forget_gate, allow_neg_eigval=True,
                                      vocab_size=max(vocab, classes), tie_word_embeddings=False, fuse_norm=False,
                                      fuse_cross_entropy=False, use_cache=False)
        self.model = GatedDeltaProductForCausalLM(cfg)

    def forward(self, tokens):
        return self.model(input_ids=tokens, use_cache=False).logits[..., :self.classes]


def build(cfg, vocab, classes):
    heads = getattr(cfg, "heads", None)
    if heads is None and cfg.d % cfg.head_dim:
        raise ValueError(f"d = {cfg.d} not divisible by head_dim {cfg.head_dim} (pass --heads)")
    return DeltaProductLM(vocab, classes, cfg.d, cfg.layers, cfg.head_dim, cfg.householder,
                          getattr(cfg, "forget_gate", False), heads, getattr(cfg, "ff", None))


if __name__ == "__main__":
    import trainer
    trainer.main("deltaproduct", build, add_args, amp=AMP)
