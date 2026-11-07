"""HyenaDNA language model from its authors' Hugging Face code (third_party/hyenadna), trained from scratch."""
from .common import LibraryLM, dims, shape

DEFAULTS = {"d_model": 128, "n_layers": 2}
AMP = None  # trained in fp32
# LongSafari/hyenadna-tiny-1k-seqlen-hf config.json, except the sizes set in build
HYENADNA_TINY = dict(emb_dim=5, filter_order=64, short_filter_order=3, hyena_order=2, activation_freq=10,
                     num_inner_mlps=2, embed_dropout=0.1, hyena_dropout=0.0, hyena_filter_dropout=0.0,
                     layer_norm_epsilon=1e-5, initializer_range=0.02, pad_vocab_size_multiple=8, train_freq=True,
                     use_bias=True, tie_word_embeddings=False)
SPECIAL_POSITIONS = 2  # the HyenaDNA configs reserve two positions for special tokens


def build(config):
    from .third_party.hyenadna.configuration_hyena import HyenaConfig
    from .third_party.hyenadna.modeling_hyena import HyenaDNAForCausalLM
    vocab, classes, max_len = dims(config["task"])
    s = shape(config, DEFAULTS)
    cfg = HyenaConfig(vocab_size=max(vocab, classes), d_model=s["d_model"], n_layer=s["n_layers"],
                      d_inner=4 * s["d_model"], max_seq_len=max_len + SPECIAL_POSITIONS, **HYENADNA_TINY)
    return LibraryLM(HyenaDNAForCausalLM(cfg), classes)
