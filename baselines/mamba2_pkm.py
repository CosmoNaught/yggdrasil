"""Mamba-2 language model of mamba_ssm with one product-key memory, the HashingMemory of Lample et al. (2019) in xlm."""
import argparse

from torch import nn

from . import mamba2
from .common import shape

DEFAULTS = {**mamba2.DEFAULTS, "pkm_keys": 256, "pkm_heads": 4, "pkm_topk": 32, "pkm_qdim": 64}
AMP = None  # trained in fp32


class MemoryBlock(nn.Module):
    """A residual block of the mamba_ssm backbone around the memory: (hidden, residual) -> (memory(RMSNorm(hidden +
    residual)), hidden + residual), the residual stream in fp32 as in the Mamba-2 blocks."""

    def __init__(self, d_model, memory):
        super().__init__()
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
        self.norm, self.mixer = RMSNorm(d_model), memory

    def forward(self, hidden_states, residual=None, inference_params=None, **kwargs):
        residual = (hidden_states + residual if residual is not None else hidden_states).float()
        return self.mixer(self.norm(residual.to(self.norm.weight.dtype))), residual


def memory_params(s):
    """xlm parameters of the memory: its argument defaults, with the sizes of s. pkm_keys sub-keys per half (pkm_keys^2
    values of width d_model), pkm_heads heads that each read pkm_topk values, and a linear query with BatchNorm of
    width pkm_qdim, split over the heads."""
    from xlm.model.memory import HashingMemory
    parser = argparse.ArgumentParser()
    HashingMemory.register_args(parser)
    if s["pkm_qdim"] % (2 * s["pkm_heads"]):
        raise ValueError(f"pkm_qdim = {s['pkm_qdim']} must split into pkm_heads = {s['pkm_heads']} even halves")
    params = parser.parse_args([f"--mem_n_keys={s['pkm_keys']}", f"--mem_heads={s['pkm_heads']}",
                                f"--mem_knn={s['pkm_topk']}", f"--mem_k_dim={s['pkm_qdim'] // s['pkm_heads']}",
                                "--mem_query_batchnorm=true"])
    HashingMemory.check_params(params)
    return params


def build(config):
    from xlm.model.memory import HashingMemory
    s = shape(config, DEFAULTS)
    lm = mamba2.build(config)
    memory = HashingMemory.build(s["d_model"], s["d_model"], memory_params(s))
    after = max(1, s["n_layers"] // 2)  # after the middle Mamba-2 layer
    layers = list(lm.model.backbone.layers)
    lm.model.backbone.layers = nn.ModuleList(layers[:after] + [MemoryBlock(s["d_model"], memory)] + layers[after:])
    return lm
