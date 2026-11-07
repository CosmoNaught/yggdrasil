"""Hyperparameters, parameter keys and the aliases that read released checkpoints (no jax import)."""
import argparse
import dataclasses
from pathlib import Path

EXPERT_TYPES = ("transition", "output")  # "output": routed experts store no transition (A = 0)
ROUTER_INPUTS = ("token_state", "token")  # "token": the router reads the token only
RMS_EPS = 1e-6  # rms(h) = h / sqrt(mean(h^2) + RMS_EPS) of the state below


@dataclasses.dataclass(frozen=True)
class Config:
    # data
    task: str = "enwik8"
    seq_len: int = 256  # training window (text tasks)
    val_windows: int = 512
    test_tokens: int = 5_000_000  # scored in windows of seq_len
    # model
    n_layers: int = 2  # token layer + routed layers
    d_state: int = 128
    block_size: int = 16  # rotation block size
    router_depth: int = 8  # 2^router_depth experts per routed layer
    router_token_dim: int = 16
    input_rank: int = None  # None: d_state / 4
    readout_rank: int = None  # None: d_state / 8
    layer_input_rank: int = None  # None: d_state / 4; 0: no input from the layer below
    expert_type: str = "transition"
    router_input: str = "token_state"
    rotation_init_std: float = 0.05  # first rotation block; the last block has std 1
    # routing
    router_topk: int = 8
    router_noise: float = 0.05  # logistic noise on router scores in training
    router_temperature: float = 0.5  # straight-through temperature
    # optimisation
    seed: int = 0
    steps: int = 40_000
    batch: int = 256
    lr: float = 1e-3
    routed_lr_scale: float = 1.0
    warmup_frac: float = 0.1
    final_lr_frac: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 4_000
    # output
    out: str = "results"
    tag: str = ""
    save: bool = True
    wandb: bool = False

    def __post_init__(self):
        for name, div in (("input_rank", 4), ("readout_rank", 8), ("layer_input_rank", 4)):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.d_state // div)
        if self.d_state % self.block_size:
            raise ValueError(f"d_state = {self.d_state} is not a multiple of block_size = {self.block_size}")
        if self.expert_type not in EXPERT_TYPES:
            raise ValueError(f"expert_type = {self.expert_type!r}, expected one of {EXPERT_TYPES}")
        if self.router_input not in ROUTER_INPUTS:
            raise ValueError(f"router_input = {self.router_input!r}, expected one of {ROUTER_INPUTS}")
        if self.router_input == "token" and self.router_token_dim == self.d_state:
            # checkpoints carry no config: readers tell the router inputs apart by the router weight width
            raise ValueError("router_input = 'token' needs router_token_dim != d_state")


# Parameter keys and shapes per layer. E = experts: vocab (token layer) or 2^router_depth (routed layer).
PARAMS = {
    "token_embed": "(vocab, input_rank)",
    "input_proj": "(E, d_state, input_rank)",
    "bias": "(E, d_state)",
    "readout": "(E, readout_rank, d_state)",
    "unembed": "(classes, readout_rank)",
    "rotation": "(E, d_state / block_size, block_size, block_size) generator G; A uses expm((G - G^T) / 2)",
    "decay": "(E, d_state), clipped to [-1, 1]",
    "router_embed": "(vocab, router_token_dim), routed layers",
    "router_weight": "(2^router_depth - 1, router_token_dim [+ d_state]), routed layers",
    "router_bias": "(2^router_depth - 1,), routed layers",
    "layer_input_down": "(layer_input_rank, d_state), routed layers",
    "layer_input_proj": "(E, d_state, layer_input_rank), routed layers",
    "hre_levels": "in training, per expert key: [(2^d, ...) for d = 0..router_depth]",
}
TOP_KEYS = ("token_layer", "routed_layers", "output_bias")  # output_bias: (classes,)
EXPERT_KEYS = ("input_proj", "bias", "readout", "rotation", "decay", "layer_input_proj")  # held as HRE in training

# Released checkpoints store these old keys (top-level, per-layer and hre_levels entries): old -> new.
KEY_ALIASES = {
    "root": "token_layer", "branches": "routed_layers", "head_b": "output_bias", "levels": "hre_levels",
    "v_tok": "token_embed", "u_in": "input_proj", "c": "bias", "p_out": "readout", "q_out": "unembed",
    "skew": "rotation", "d_raw": "decay", "tok": "router_embed", "w_node": "router_weight", "b_node": "router_bias",
    "v_low": "layer_input_down", "u_low": "layer_input_proj",
}
# Released run records store these old config fields: old -> new. "branch" maps by value (VARIANT_ALIASES).
FIELD_ALIASES = {
    "T": "seq_len", "test_chars": "test_tokens", "layers": "n_layers", "n": "d_state", "block": "block_size",
    "depth": "router_depth", "feat": "router_token_dim", "rank": "input_rank", "out_rank": "readout_rank",
    "lower_rank": "layer_input_rank", "skew_std": "rotation_init_std", "beam": "router_topk",
    "route_noise": "router_noise", "route_tau": "router_temperature", "branch_lr": "routed_lr_scale",
}
VARIANT_ALIASES = {"dyn": {}, "out": {"expert_type": "output"}, "tok": {"router_input": "token"}}


def _rename(key):
    return "/".join(KEY_ALIASES.get(part, part) for part in key.split("/"))


def upgrade_keys(params):
    """Old checkpoint keys -> new, in nested dicts/lists or flat "a/b/c" keys; new keys pass through."""
    if isinstance(params, dict):
        return {_rename(k): upgrade_keys(v) for k, v in params.items()}
    if isinstance(params, (list, tuple)):
        return type(params)(upgrade_keys(v) for v in params)
    return params


def upgrade_config(fields):
    """Old run-record config fields -> new; unknown fields pass through."""
    out = {FIELD_ALIASES.get(k, k): v for k, v in fields.items() if k != "branch"}
    if "branch" in fields:
        out.update(VARIANT_ALIASES[fields["branch"]])
    return out


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


def parse(argv=None):
    """--config <yaml> plus --<field> overrides -> Config."""
    import yaml

    types = {f.name: _flag if f.type is bool else f.type for f in dataclasses.fields(Config)}
    ap = argparse.ArgumentParser(prog="python -m yggdrasil.train")
    ap.add_argument("--config", type=Path)
    for name, t in types.items():
        ap.add_argument(f"--{name}", type=t, default=None)
    args = vars(ap.parse_args(argv))
    path = args.pop("config")
    base = yaml.safe_load(path.read_text()) if path else {}
    unknown = set(base) - set(types)
    if unknown:
        ap.error(f"unknown keys in {path}: {sorted(unknown)}")
    base = {k: types[k](v) for k, v in base.items()}  # YAML reads 1e-3 as a string
    return Config(**{**base, **{k: v for k, v in args.items() if v is not None}})
