"""Training, evaluation and checkpoints.

    python -m yggdrasil.train --config configs/enwik8/medium.yaml --seed 1

Under a W&B sweep (WANDB_SWEEP_ID set) or with --wandb true, the run logs to W&B and its summary holds "objective".
"""
import dataclasses
import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import KEY_ALIASES, parse
from .data import build_task
from .experts import materialize, to_hre
from .model import forward, init, save

MAX_NONFINITE = 1000  # consecutive skipped non-finite updates before optax gives up skipping
FAILED_SCORE = -99.0  # sweep objective of a text run without a finite validation score
STEP_KEY_STRIDE = 1_000_003  # step keys PRNGKey(seed * stride + step) stay distinct across seeds


def run_name(cfg):
    variant = "" if cfg.expert_type == "transition" else f"_{cfg.expert_type}"
    variant += "" if cfg.router_input == "token_state" else f"_{cfg.router_input}"
    size = f"d{cfg.d_state}_block{cfg.block_size}_depth{cfg.router_depth}_top{cfg.router_topk}"
    return f"{cfg.task}_{size}{variant}{cfg.tag}_s{cfg.seed}"


_RELEASED_NAME = {new: old for old, new in KEY_ALIASES.items()}


def _released_path(path):
    return tuple(_RELEASED_NAME.get(k.key, k.key) if hasattr(k, "key") else k.idx for k in path)


def global_norm(tree):
    """L2 norm over all arrays, summed in the order of the released checkpoint key names.

    Float addition is not associative: this order makes training repeat the released runs bit for bit.
    """
    arrays = sorted(jax.tree_util.tree_flatten_with_path(tree)[0], key=lambda kv: _released_path(kv[0]))
    return optax.tree.norm([v for _, v in arrays])


def clip_by_global_norm(max_norm):
    """optax.clip_by_global_norm with global_norm above."""
    def update(updates, state, params=None):
        g_norm = global_norm(updates)
        keep = jnp.squeeze(g_norm < max_norm)
        return jax.tree_util.tree_map(lambda t: jax.lax.select(keep, t, (t / g_norm.astype(t.dtype)) * max_norm),
                                      updates), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


def build_optimizer(cfg):
    """Adam with global-norm clipping, linear warm-up and cosine decay; skips non-finite updates."""
    warmup = round(cfg.warmup_frac * cfg.steps)
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warmup, cfg.steps, cfg.lr * cfg.final_lr_frac)
    return optax.apply_if_finite(optax.chain(clip_by_global_norm(cfg.grad_clip), optax.adam(sched)), MAX_NONFINITE)


def make_train_step(cfg, opt):
    """-> jitted step(params, opt_state, x, y, m, key) -> params, opt_state, loss, gradient norm before clipping."""
    def loss_fn(p, x, y, m, key):
        logp, _ = forward(p, x, cfg, "st", key)
        return -(jnp.take_along_axis(logp, y[..., None], -1)[..., 0] * m / m.sum()).sum()

    def step(p, s, x, y, m, key):
        loss, g = jax.value_and_grad(loss_fn)(p, x, y, m, key)
        u, s = opt.update(g, s, p)
        if cfg.routed_lr_scale != 1.0:
            routed = jax.tree_util.tree_map(lambda v: v * cfg.routed_lr_scale, u["routed_layers"])
            u = dict(u, routed_layers=routed)
        return optax.apply_updates(p, u), s, loss, global_norm(g)
    return jax.jit(step)


def init_params(cfg, vocab, classes):
    """-> params in training form: routed experts as hierarchical residual experts (to_hre)."""
    params = init(jax.random.PRNGKey(cfg.seed), cfg, vocab, classes)
    return dict(params, routed_layers=[to_hre(p, cfg.router_depth) for p in params["routed_layers"]])


def stored_params(cfg, vocab, classes):
    """Parameter count with per-expert arrays, as checkpoints store them."""
    shapes = jax.eval_shape(lambda k: init(k, cfg, vocab, classes), jax.random.PRNGKey(0))
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(shapes)))


def fit(cfg, task):
    """Yields (step, params, opt_state, loss, gradient norm) after each of cfg.steps updates."""
    params = init_params(cfg, task.vocab, task.classes)
    opt = build_optimizer(cfg)
    opt_state = opt.init(params)
    step = make_train_step(cfg, opt)
    rng = np.random.default_rng(cfg.seed)
    for it in range(1, cfg.steps + 1):
        x, y, m = task.batch(rng, cfg.batch)
        key = jax.random.PRNGKey(cfg.seed * STEP_KEY_STRIDE + it)
        params, opt_state, loss, gnorm = step(params, opt_state, x, y, m, key)
        yield it, params, opt_state, loss, gnorm


@functools.partial(jax.jit, static_argnames="cfg")
def _predict(p, x, cfg):
    return forward(p, x, cfg)[0].argmax(-1)


@functools.partial(jax.jit, static_argnames="cfg")
def _nll(p, x, y, cfg):
    return -jnp.take_along_axis(forward(p, x, cfg)[0], y[..., None], -1).sum()


def evaluate(cfg, params, split, task=None):
    """Greedy routing -> bits per token (text) or {"scaled", "acc", "chance", "n"} (state tasks)."""
    task = task or build_task(cfg)
    if task.kind == "text":
        return task.evaluate(lambda x, y: _nll(params, x, y, cfg), split)
    return task.evaluate(lambda x: _predict(params, x, cfg), split)


def validate(cfg, task, params):
    """-> (record fields, score); a higher score is better."""
    if task.kind == "text":
        bits = evaluate(cfg, params, "val", task)
        return {"val_bits": bits}, -bits  # bits per byte (enwik8) or per base (hg38)
    metric = evaluate(cfg, params, "val", task)
    return {"val_metric": metric}, metric["scaled"]  # scaled final-answer accuracy, can be negative


def finalize(cfg, task, best, out):
    """Test score and per-length curve of the best checkpoint; writes <out>.npz."""
    res = {}
    if "params" in best:
        p = jax.device_put(best["params"])
        res["test"] = evaluate(cfg, p, "test", task)
        if task.kind == "state":
            res["test_curve"] = task.curve(lambda x: _predict(p, x, cfg))
        if cfg.save:
            save(out.with_suffix(".npz"), materialize(best["params"]))
    return res


def _peak_gb():
    try:
        return round(jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9, 3)
    except (AttributeError, KeyError, TypeError):  # no memory statistics on this backend
        return None


def _code_md5():
    h = hashlib.md5()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def _start_wandb(cfg, name, n_params):
    """W&B run under a sweep (WANDB_SWEEP_ID) or with --wandb true, else None."""
    if not (cfg.wandb or "WANDB_SWEEP_ID" in os.environ):
        return None
    import wandb
    return wandb.init(name=os.environ.get("WANDB_NAME", name), config=dict(dataclasses.asdict(cfg), n_params=n_params))


def _finish_wandb(wb, best, res, ckpt):
    objective = best["score"] if math.isfinite(best["score"]) else FAILED_SCORE
    wb.summary.update({"objective": objective, "best_val": res["best_val"], "test": res.get("test"), "ckpt": ckpt})
    wb.finish()


def train(cfg):
    """Train, keep the best validation checkpoint, score it on test; writes <out>/<name>.json and .npz."""
    task = build_task(cfg)
    text = task.kind == "text"
    n_params = stored_params(cfg, task.vocab, task.classes)
    name = run_name(cfg)
    out = Path(cfg.out) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = _start_wandb(cfg, name, n_params)
    print(name, n_params, jax.devices()[0], flush=True)
    log, t0, best = [], time.time(), {"score": -math.inf}
    for it, params, _, loss, gnorm in fit(cfg, task):
        if it % cfg.eval_every and it != cfg.steps:
            continue
        rec, score = validate(cfg, task, params)
        rec = {"step": it, "loss": float(loss), "grad_norm": float(gnorm), "time_s": round(time.time() - t0, 1), **rec}
        if text:
            rec["tokens"] = it * cfg.batch * cfg.seq_len
        if score > best["score"]:
            best = {"score": score, "step": it, "params": jax.device_get(params)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if wb is not None:
            wb.log({"loss": rec["loss"], "val": -score if text else score}, step=it)
    del params  # room for the best parameters on the device
    res = {"config": dataclasses.asdict(cfg), "name": name, "n_params": n_params, "log": log,
           "best_step": best.get("step"), "best_val": -best["score"] if text else best["score"],
           "train_s": round(time.time() - t0, 1), "code_md5": _code_md5(),
           "device": str(jax.devices()[0].device_kind), "peak_mem_gb": _peak_gb(), **finalize(cfg, task, best, out)}
    out.with_suffix(".json").write_text(json.dumps(res))
    if wb is not None:
        _finish_wandb(wb, best, res, str(out.with_suffix(".npz")))
    print("DONE", name, "best val", res["best_val"], "test", res.get("test"), flush=True)
    return res


if __name__ == "__main__":
    train(parse())
