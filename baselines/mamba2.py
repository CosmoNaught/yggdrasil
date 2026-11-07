"""Mamba-2 language model of mamba_ssm."""
from .common import LibraryLM, dims, shape

DEFAULTS = {"d_model": 128, "n_layers": 2, "d_state": 64, "head_dim": 16}
AMP = None  # trained in fp32
EXPAND = 2  # mamba_ssm defaults
CONV_WIDTH = 4


def build(config):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    vocab, classes, _ = dims(config["task"])
    s = shape(config, DEFAULTS)
    d_inner = EXPAND * s["d_model"]
    if d_inner % s["head_dim"] or s["head_dim"] > s["d_model"] // 4:
        raise ValueError(f"head_dim = {s['head_dim']} must divide {d_inner} and be at most d_model / 4")
    cfg = MambaConfig(d_model=s["d_model"], d_intermediate=0, n_layer=s["n_layers"], vocab_size=max(vocab, classes),
                      ssm_cfg={"layer": "Mamba2", "d_state": s["d_state"], "headdim": s["head_dim"],
                               "expand": EXPAND, "d_conv": CONV_WIDTH},
                      rms_norm=True, residual_in_fp32=True, fused_add_norm=True, pad_vocab_size_multiple=8,
                      tie_embeddings=False)
    return LibraryLM(MambaLMHeadModel(cfg), classes)
