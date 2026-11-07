"""Llama decoder of transformers: pre-RMSNorm, rotary positions, SwiGLU MLP, no biases, untied head."""
import math

import torch

from .common import LibraryLM, dims, shape

DEFAULTS = {"d_model": 128, "n_layers": 4, "head_dim": 32}
AMP = torch.bfloat16  # trained under bf16 autocast


def ffn_width(d_model):
    """SwiGLU width 8 d_model / 3, rounded up to a multiple of 32 (the Llama rule)."""
    return math.ceil(8 * d_model / 3 / 32) * 32


def attention():
    """FlashAttention-2 when it is installed and a GPU is present, else PyTorch SDPA (the same model)."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2" if torch.cuda.is_available() else "sdpa"


def build(config):
    from transformers import LlamaConfig, LlamaForCausalLM
    vocab, classes, max_len = dims(config["task"])
    s = shape(config, DEFAULTS)
    d = s["d_model"]
    if d % s["head_dim"]:
        raise ValueError(f"d_model = {d} is not divisible by head_dim = {s['head_dim']}")
    heads = d // s["head_dim"]
    cfg = LlamaConfig(vocab_size=max(vocab, classes), hidden_size=d, intermediate_size=ffn_width(d),
                      num_hidden_layers=s["n_layers"], num_attention_heads=heads, num_key_value_heads=heads,
                      head_dim=s["head_dim"], max_position_embeddings=max_len, rms_norm_eps=1e-5, rope_theta=1e4,
                      attention_bias=False, mlp_bias=False, attention_dropout=0.0, tie_word_embeddings=False,
                      initializer_range=0.02, use_cache=False)
    model = LlamaForCausalLM._from_config(cfg, attn_implementation=attention(), dtype=torch.float32)
    return LibraryLM(model, classes, use_cache=False)
