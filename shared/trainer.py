"""Shared training loop of the baselines (the protocol of models/yggdrasil/training/train.py). Each models/<model>/training/train.py defines
build(cfg, vocab, classes) and add_args(parser) and calls main(<model>, build, add_args).

    python models/<model>/training/train.py --task <task> <size flags> --lr <lr> --seed <seed> [--wandb true]
    python models/<model>/training/train.py --task <task> --seeds <s1,s2,s3> ...         one W&B sweep trial


Data and evaluation (models/yggdrasil/training/train.py, through tasks.py, a verbatim copy of models/yggdrasil/training/train.py's task code):
  enwik8, hg38   LMTask: enwik8 raw bytes split 90M / 5M / 5M (vocab 256); hg38 data/hg38/{train,valid,test}.u8
                 (vocab 5). Batches of --batch random windows of --T (numpy default_rng(seed)), zero state per window.
                 Validation bits per token on the first --val_windows windows, test on the first --test_chars // T.
  state tasks    StateTask: s5_word, a5_word (train 32, test 500), parity_check, cycle_navigation, modular_arithmetic
                 (train 40, test 256). Loss on every labelled prefix; validation = sequence accuracy at the test
                 length on 512 sequences (seed 20000 + s); test at L in {32, 64, 128, 256, 500} (seed 10000 + s).
Optimiser: Adam(0.9, --beta2, 1e-8) (AdamW with decay on >= 2-d weights when --weight_decay > 0), global-norm clip 1,
optax warmup_cosine_decay_schedule(0, lr, warmup, steps, lr / 100) with warmup = min(1000, steps // 10) or
--warmup_frac * steps, non-finite updates skipped (the schedule counts applied updates). fp32 with TF32 off, unless
the model sets AMP (bf16 autocast, fp32 weights and optimiser state). The best-validation weights are kept and tested.

Output in <--out>/<name>/:
  ckpt.pt       torch.save dict: model (folder name), task, vocab, classes, config (all flags, a dict),
                state_dict (best-validation weights, fp32, CPU), amp (None or "bfloat16": the autocast dtype of
                training and evaluation), best_val, best_step, test, n_params, versions
  result.json   the same without the weights, plus the evaluation log
load(path) rebuilds the module from ckpt.pt with the model's own build() (see below).
W&B (--wandb true, or under a sweep agent): loss / val / best_val per evaluation; summary objective (sweeps maximise
it) = -best val bits (LM) or best val sequence accuracy (state), best_val, test, n_params, ckpt.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(HERE))
import tasks  # noqa: E402

LN2 = math.log(2.0)
DEFAULT_STEPS = {"s5_word": 80_000, "a5_word": 80_000, "enwik8": 10_000, "hg38": 10_000}  # others 40_000
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


class LMTask:
    """models/yggdrasil/training/train.py LMTask (enwik8 and hg38), zero state per window, bits per token."""

    def __init__(self, name, T, val_windows, test_chars):
        data = ROOT / "data"
        if name == "enwik8":
            raw = np.frombuffer((data / name).read_bytes(), np.uint8)
            self.train, val, test = raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]
        else:
            self.train, val, test = (np.memmap(data / name / f"{s}.u8", np.uint8, "r") for s in ("train", "valid", "test"))
        self.vocab = self.classes = tasks.LM_TASKS[name]
        self.T = self.train_len = self.eval_len = T
        self.streams = {"val": val, "test": test}  # raw splits (streaming evaluation)
        self.sets = {"val": self._windows(val, val_windows), "test": self._windows(test, test_chars // T)}

    def _windows(self, ids, m):
        m = min(m, (len(ids) - 1) // self.T)
        ids = np.asarray(ids[:m * self.T + 1], np.int32)
        return ids[:-1].reshape(m, self.T), ids[1:].reshape(m, self.T)

    def batch(self, rng, size):
        idx = rng.integers(0, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        return self.train[idx].astype(np.int32), self.train[idx + 1].astype(np.int32), np.ones((size, self.T), np.float32)

    def evaluate(self, nll_fn, split, chunk=64):
        x, y = self.sets[split]
        nll = sum(float(nll_fn(x[i:i + chunk], y[i:i + chunk])) for i in range(0, len(x), chunk))
        return nll / y.size / LN2


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


def parser(model, add_args):
    ap = argparse.ArgumentParser(description=f"train {model}")
    add = ap.add_argument
    add("--task", required=True, choices=[*tasks.LM_TASKS, *tasks.STATE_TASKS])
    add("--seed", type=int, default=0)
    add("--seeds", default="", help="comma list: one process per seed, W&B objective = mean (one sweep trial)")
    add("--steps", type=int, default=None, help="default 80k s5/a5, 40k other state tasks, 10k enwik8 / hg38")
    add("--batch", type=int, default=None, help="default 256 (state tasks), 64 (enwik8 / hg38)")
    add("--lr", type=float, default=1e-3)
    add("--beta2", type=float, default=0.999)
    add("--weight_decay", type=float, default=0.0)
    add("--warmup_frac", type=float, default=None, help="warmup = this fraction of steps (default min(1000, steps/10))")
    add("--T", type=int, default=256, help="LM window length")
    add("--val_windows", type=int, default=512)
    add("--test_chars", type=int, default=5_000_000)
    add("--eval_every", type=int, default=None, help="default 2000 (LM), 4000 (state tasks)")
    add("--eval_batch", type=int, default=0, help="sequences per evaluation call (default 64 LM, 128 state)")
    add("--log_every", type=int, default=100)
    add("--out", default=str(ROOT / "results" / "baselines"))
    add("--tag", default="")
    add("--wandb", type=_flag, default=False)
    add("--project", default="yggdrasil-baselines")
    add_args(ap)
    return ap


def lr_at(k, a):
    w = min(1000, a.steps // 10) if a.warmup_frac is None else int(round(a.warmup_frac * a.steps))
    if k < w:
        return a.lr * k / w
    frac = min(1.0, (k - w) / max(1, a.steps - w))
    end = a.lr * 0.01
    return end + (a.lr - end) * 0.5 * (1 + math.cos(math.pi * frac))


def versions():
    import importlib.metadata as md
    out = {"torch": torch.__version__}
    for p in ("mamba-ssm", "causal-conv1d", "flash-attn", "flash-linear-attention", "fla-core", "transformers"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def train(model_name, build, a, amp=None):
    lm = a.task in tasks.LM_TASKS
    task = LMTask(a.task, a.T, a.val_windows, a.test_chars) if lm else tasks.StateTask(a.task, a.seed)
    a.max_len = max(task.train_len, task.eval_len)  # longest sequence the model sees (position tables)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    model = build(a, task.vocab, task.classes).to(dev).float()
    n_params = sum(p.numel() for p in model.parameters())
    dec = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=0.0, betas=(0.9, a.beta2), eps=1e-8)
    C = task.classes
    name = f"{model_name}_{a.task}{a.tag}_s{a.seed}"
    sweep = "WANDB_SWEEP_ID" in os.environ
    wb = None
    if a.wandb or sweep:
        import wandb
        project = None if sweep else os.environ.get("WANDB_PROJECT", a.project)
        wb = wandb.init(project=project, name=os.environ.get("WANDB_NAME", name),
                        config=dict(vars(a), model=model_name, n_params=n_params))
        if wb.id not in name:
            name = f"{name}_{wb.id}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    print(name, "params", n_params, dev_name, flush=True)

    def fwd(x):
        with torch.autocast(dev.type, dtype=amp or torch.float32, enabled=amp is not None):
            return model(x).float()

    @torch.no_grad()
    def predict(x):
        return fwd(torch.as_tensor(x, device=dev, dtype=torch.long)).argmax(-1).cpu().numpy()

    @torch.no_grad()
    def nll(x, y):
        lg = fwd(torch.as_tensor(x, device=dev, dtype=torch.long))
        return F.cross_entropy(lg.reshape(-1, C), torch.as_tensor(y, device=dev, dtype=torch.long).reshape(-1),
                               reduction="sum").item()

    def evaluate(split):
        model.eval()
        chunk = a.eval_batch or (64 if lm else tasks.EVAL_CHUNK)
        r = task.evaluate(nll, split, chunk) if lm else task.evaluate(predict, split, chunk)
        model.train()
        return r

    rng, log, t0, k = np.random.default_rng(a.seed), [], time.time(), 0
    best = {"val": float("inf") if lm else -1.0}
    model.train()
    for it in range(1, a.steps + 1):
        x, y, m = (torch.as_tensor(v, device=dev) for v in task.batch(rng, a.batch))
        lg = fwd(x.long())
        w = m / m.sum()
        loss = (F.cross_entropy(lg.reshape(-1, C), y.long().reshape(-1), reduction="none") * w.reshape(-1)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ce = loss.item()
        if math.isfinite(ce) and torch.isfinite(gn):
            for g in opt.param_groups:
                g["lr"] = lr_at(k, a)
            opt.step()
            k += 1
        if it == 1 or it % a.log_every == 0:
            print(json.dumps({"step": it, "loss": ce}), flush=True)
            if wb is not None:
                wb.log({"loss": ce}, step=it)
        if it % a.eval_every and it != a.steps:
            continue
        rec = {"step": it, "loss": ce, "skipped": it - k, "time_s": round(time.time() - t0, 1)}
        if lm:
            rec["val_bpc"] = val = evaluate("val")
            better = val < best["val"]
        else:
            rec["val_seq"] = evaluate("val")
            val = rec["val_seq"][str(task.eval_len)]
            better = val > best["val"]
        if better:
            best = {"val": val, "step": it, "sd": {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if wb is not None:
            wb.log({"loss": ce, "val": val, "best_val": best["val"]}, step=it)

    res = {"model": model_name, "task": a.task, "vocab": task.vocab, "classes": C, "config": vars(a),
           "amp": None if amp is None else str(amp).replace("torch.", ""), "n_params": n_params,
           "best_val": best["val"], "best_step": best.get("step"), "versions": versions(),
           "device": dev_name, "train_s": round(time.time() - t0, 1)}
    if "sd" in best:
        model.load_state_dict(best["sd"])
        res["test"] = evaluate("test")
        torch.save({**res, "state_dict": best["sd"]}, out / "ckpt.pt")
    (out / "result.json").write_text(json.dumps({**res, "log": log}))
    if wb is not None:
        obj = (-best["val"] if math.isfinite(best["val"]) else -99.0) if lm else best["val"]
        wb.summary.update({"objective": obj, "best_val": best["val"], "test": res.get("test"), "n_params": n_params,
                           "ckpt": str(out / "ckpt.pt")})
        wb.finish()
    print("DONE", name, "best val", best["val"], "test", res.get("test"), "ckpt", out / "ckpt.pt", flush=True)


def trial(a, argv):
    """One W&B sweep trial: one process per seed on this GPU; objective = mean over seeds of the best validation value
    (state: sequence accuracy at the test length, a failed seed counts 0; LM: -bits per token)."""
    import wandb
    wb = wandb.init()
    seeds = a.seeds.split(",")
    rest, skip = [], False  # argv without --seeds and --seed
    for x in argv:
        if skip or x.split("=")[0] in ("--seeds", "--seed"):
            skip = x in ("--seeds", "--seed")
            continue
        rest.append(x)
    keep = ("WANDB_API_KEY", "WANDB_SILENT", "WANDB_MODE", "WANDB_ENTITY", "WANDB_DIR", "WANDB_BASE_URL")
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_") or k in keep}
    env.update(WANDB_PROJECT=wb.project, WANDB_RUN_GROUP=f"trial_{wb.id}")
    out = Path(a.out) / wb.id
    out.mkdir(parents=True, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0]), *rest, "--seed", s, "--out", str(out),
                               "--wandb", "true"], env=dict(env, WANDB_NAME=f"trial_{wb.id}_{a.task}_s{s}"),
                              stdout=open(out / f"s{s}.log", "w"), stderr=subprocess.STDOUT) for s in seeds]
    for q in procs:
        q.wait()
    vals = {}
    for s in seeds:
        hits = list(out.glob(f"*_s{s}_*/result.json"))
        vals[f"{a.task}/s{s}"] = json.loads(hits[0].read_text())["best_val"] if hits else None
    ok = [v for v in vals.values() if v is not None]
    lm = a.task in tasks.LM_TASKS
    score = (-sum(ok) / len(ok) if lm else sum(ok) / len(seeds)) if ok else (-99.0 if lm else 0.0)
    wb.summary.update({"objective": score, **vals})
    wb.finish()


def main(model_name, build, add_args, amp=None):
    a = parser(model_name, add_args).parse_args()
    lm = a.task in tasks.LM_TASKS
    a.steps = a.steps or DEFAULT_STEPS.get(a.task, 40_000)
    a.batch = a.batch or (64 if lm else 256)
    a.eval_every = a.eval_every or (2_000 if lm else 4_000)
    if a.seeds:
        trial(a, sys.argv[1:])
    else:
        train(model_name, build, a, amp)


def load(path, device="cpu"):
    """ckpt.pt -> (module with the best weights, eval mode, the checkpoint dict). The module is built by
    models/<model>/training/train.py build(config, vocab, classes) in the current Python (the model's package must be
    importable: use the model's environment, env.py)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = importlib.util.spec_from_file_location(f"train_{ck['model']}", MODELS / ck["model"] / "training" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # train.py runs main() only under __main__
    model = mod.build(types.SimpleNamespace(**ck["config"]), ck["vocab"], ck["classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
