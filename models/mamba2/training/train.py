"""Mamba-2 language model on mamba_ssm 2.3.0 (github.com/state-spaces/mamba, Apache-2.0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch.nn as nn  # noqa: E402


def add_args(ap):
    # Requires 2 * d % head_dim == 0, (2 * d + 2 * state) % 8 == 0, head_dim <= d / 4
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--state", type=int, default=64, help="d_state")
    ap.add_argument("--head_dim", type=int, default=16, help="headdim (heads = 2 d / head_dim)")
    ap.add_argument("--dt_max", type=float, default=0.1, help="dt initialisation range upper end (package default)")


class Mamba2LM(nn.Module):
    def __init__(self, vocab, classes, d, layers, state, head_dim, dt_max=0.1):
        super().__init__()
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        self.classes = classes
        cfg = MambaConfig(d_model=d, d_intermediate=0, n_layer=layers, vocab_size=max(vocab, classes),
                          ssm_cfg={"layer": "Mamba2", "d_state": state, "headdim": head_dim, "expand": 2,
                                   "d_conv": 4, "dt_max": dt_max},
                          rms_norm=True, residual_in_fp32=True, fused_add_norm=True, pad_vocab_size_multiple=8,
                          tie_embeddings=False)
        self.model = MambaLMHeadModel(cfg)

    def forward(self, tokens):
        return self.model(tokens).logits[..., :self.classes]  # drop vocab padding


def build(cfg, vocab, classes):
    return Mamba2LM(vocab, classes, cfg.d, cfg.layers, cfg.state, cfg.head_dim, getattr(cfg, "dt_max", 0.1))


if __name__ == "__main__":
    import trainer
    trainer.main("mamba2", build, add_args)
