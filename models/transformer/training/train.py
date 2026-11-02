"""Llama-style decoder baseline: pre-RMSNorm, RoPE, SwiGLU, untied head."""
# Upstream: transformers 4.57.6 LlamaForCausalLM (Apache-2.0) with flash-attn 2.8.3.post2 (BSD-3-Clause), unmodified.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("flashattn")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

AMP = torch.bfloat16


def add_args(ap):
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--attn", default="flash_attention_2", choices=("flash_attention_2", "sdpa", "eager"),
                    help="attention kernel (flash_attention_2 for training; the others give the same model)")


class Transformer(nn.Module):
    def __init__(self, vocab, classes, d, layers, head_dim, max_len, attn="flash_attention_2"):
        super().__init__()
        from transformers import LlamaConfig, LlamaForCausalLM
        self.classes = classes
        cfg = LlamaConfig(vocab_size=max(vocab, classes), hidden_size=d, intermediate_size=math.ceil(8 * d / 3 / 32) * 32,
                          num_hidden_layers=layers, num_attention_heads=d // head_dim,
                          num_key_value_heads=d // head_dim, head_dim=head_dim, max_position_embeddings=max_len,
                          rms_norm_eps=1e-5, rope_theta=1e4, attention_bias=False, mlp_bias=False,
                          attention_dropout=0.0, tie_word_embeddings=False, initializer_range=0.02, use_cache=False)
        self.model = LlamaForCausalLM._from_config(cfg, attn_implementation=attn, dtype=torch.float32)

    def forward(self, tokens):
        return self.model(input_ids=tokens, use_cache=False).logits[..., :self.classes]


def build(cfg, vocab, classes):
    if cfg.d % cfg.head_dim:
        raise ValueError(f"d = {cfg.d} not divisible by head_dim {cfg.head_dim}")
    return Transformer(vocab, classes, cfg.d, cfg.layers, cfg.head_dim, cfg.max_len,
                       getattr(cfg, "attn", "flash_attention_2"))


if __name__ == "__main__":
    import trainer
    trainer.main("transformer", build, add_args, amp=AMP)
