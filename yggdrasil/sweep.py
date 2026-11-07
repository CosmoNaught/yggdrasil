"""W&B sweep of one configuration: 8 Bayesian trials of 10k steps, each one run of yggdrasil.train.

    python -m yggdrasil.sweep --config configs/enwik8/medium.yaml [--project yggdrasil] [--dry]
    wandb agent <entity>/<project>/<sweep id>      # runs the trials; this module only creates the sweep
"""
import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from .config import parse

TRIALS = 8
TRIAL = {"text": {"steps": 10_000, "eval_every": 2_000, "test_tokens": 500_000, "seed": 0},
         "state": {"steps": 10_000, "eval_every": 2_000, "seed": 160}}
SEARCH = {"lr": ("log_uniform", 5e-4, 2e-2), "routed_lr_scale": [0.3, 1.0], "router_noise": [0.0, 0.05, 0.1],
          "router_topk": [4, 8]}
STATE_SIZE = {"n_layers": [2, 3, 5], "d_state": [64, 128, 256]}  # 1, 2 or 4 routed layers; ranks follow d_state
TEXT_TASKS = ("enwik8", "hg38")


def _param(v):
    if isinstance(v, list):
        return {"values": v}
    if isinstance(v, tuple):
        return {"distribution": "log_uniform_values", "min": v[1], "max": v[2]}
    return {"value": v}


def spec(cfg):
    """Config -> W&B sweep spec; the trial's objective is the summary "objective" that train() writes."""
    kind = "text" if cfg.task in TEXT_TASKS else "state"
    search = dict(SEARCH, **(STATE_SIZE if kind == "state" else {}))
    fixed = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in search and k not in ("save", "wandb")}
    if kind == "state":  # ranks follow the searched d_state; the text-only fields do not apply
        for k in ("input_rank", "layer_input_rank", "readout_rank", "seq_len", "val_windows", "test_tokens"):
            fixed.pop(k, None)
    fixed.update(TRIAL[kind])
    return {"name": cfg.task, "method": "bayes", "metric": {"name": "objective", "goal": "maximize"},
            "run_cap": TRIALS, "command": ["${env}", "${interpreter}", "-m", "yggdrasil.train", "${args}"],
            "parameters": {k: _param(v) for k, v in {**fixed, **search}.items()}}


def main():
    ap = argparse.ArgumentParser(prog="python -m yggdrasil.sweep")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--project", default="yggdrasil")
    ap.add_argument("--dry", action="store_true", help="print the spec only")
    a = ap.parse_args()
    s = spec(parse(["--config", str(a.config)]))
    if a.dry:
        print(yaml.safe_dump(s, sort_keys=False))
        return
    import wandb
    print(json.dumps({"sweep": wandb.sweep(s, project=a.project), "project": a.project}))


if __name__ == "__main__":
    main()
