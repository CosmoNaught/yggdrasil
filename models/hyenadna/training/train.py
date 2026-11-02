"""HyenaDNA training (from scratch).

Source: the HyenaDNA authors' Hugging Face port of their model (Nguyen et al., "HyenaDNA: Long-Range Genomic Sequence
Modeling at Single Nucleotide Resolution", NeurIPS 2023; https://github.com/HazyResearch/hyena-dna at commit
d553021b483b82980aa4b868b37ec2d4332e198a), files modeling_hyena.py and configuration_hyena.py of
https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf at revision e8c1effa8673814e257e627d2e1eda9ea5a373f6
(BSD-3-Clause, model card in third_party/hyenadna_hf/MODEL_CARD.md; the same files in the small-32k and medium-160k
repos), vendored unmodified in third_party/hyenadna_hf/. Long convolutions by torch.fft (the port has no fused
kernel). Used: HyenaDNAForCausalLM from HyenaConfig, trained from scratch (no pretrained weights); only the training
loop (shared/trainer.py) is ours.
Settings: d_model --d, --layers, d_inner 4 d, Hyena order 2, filter order --filter_order (64), short filter 3,
positional embedding size 5, filter sine frequency 10, embedding dropout 0.1 (all as the tiny-1k checkpoint),
max_seq_len = the longest training or evaluation sequence + 2.

    python models/hyenadna/training/train.py --task <task> --d <width> --layers <layers> --filter_order <order> --lr <lr> --seed <seed>

"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

import torch.nn as nn  # noqa: E402

TP = Path(__file__).resolve().parent / "third_party"


def add_args(ap):
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--filter_order", type=int, default=64)


class HyenaDNALM(nn.Module):
    def __init__(self, vocab, classes, d, layers, max_len, filter_order=64):
        super().__init__()
        if str(TP) not in sys.path:
            sys.path.insert(0, str(TP))
        conf = importlib.import_module("hyenadna_hf.configuration_hyena")
        mod = importlib.import_module("hyenadna_hf.modeling_hyena")
        self.classes = classes
        cfg = conf.HyenaConfig(vocab_size=max(vocab, classes), d_model=d, n_layer=layers, d_inner=4 * d,
                               max_seq_len=max_len + 2, emb_dim=5, filter_order=filter_order, short_filter_order=3,
                               hyena_order=2, activation_freq=10, num_inner_mlps=2, embed_dropout=0.1,
                               hyena_dropout=0.0, hyena_filter_dropout=0.0, layer_norm_epsilon=1e-5,
                               initializer_range=0.02, pad_vocab_size_multiple=8, train_freq=True, use_bias=True,
                               tie_word_embeddings=False)
        self.model = mod.HyenaDNAForCausalLM(cfg)

    def forward(self, tokens):
        return self.model(input_ids=tokens).logits[..., :self.classes]


def build(cfg, vocab, classes):
    return HyenaDNALM(vocab, classes, cfg.d, cfg.layers, cfg.max_len, getattr(cfg, "filter_order", 64))


if __name__ == "__main__":
    import trainer
    trainer.main("hyenadna", build, add_args)
