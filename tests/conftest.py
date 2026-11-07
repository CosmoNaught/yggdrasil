"""Shared fixtures, the adapters from the tests to the package, and the comparisons.

The goldens in tests/golden/ are outputs of the reported checkpoints and of fixed-seed models: CPU goldens from an AMD
EPYC (x86-64, AVX2, no AVX-512), GPU goldens from an NVIDIA RTX A5000; each GPU golden names its GPU. Every call from
a test into the package goes through one adapter function below.

Paths (environment variables):
    YGGDRASIL_DATA  data directory with enwik8 and hg38/{train,valid,test}.u8 (default <repo>/data, tasks.text.DATA)
    YGG_RUNS        directory with the reported checkpoints <id>.npz and their run records <id>.json, one per id of
                    golden/checkpoints.json (default <repo>/results)

Markers: gpu (needs a GPU; skipped without one), slow (full test split or 200 training steps), data (needs
YGGDRASIL_DATA), ckpt (needs YGG_RUNS).
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")  # fp32 matmuls without TF32, as the goldens ran
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.text import DATA  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"
RUNS = Path(os.environ.get("YGG_RUNS", REPO / "results"))

# XLA's CPU code depends on the instruction set: on an AVX-512 CPU (Intel Xeon Gold 6426Y) fp32 log-probabilities
# move by up to ~2e-5 relative after 256 recurrent steps. On a CPU of the goldens' kind (AVX2 without AVX-512) the
# tests compare log-probabilities tightly, otherwise probabilities (see logp_close) and shorter traces.


def golden(name):
    return json.loads((GOLDEN / name).read_text())


def cpu_flags():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


_FLAGS = cpu_flags()
SAME_CPU = platform.machine() == "x86_64" and "avx2" in _FLAGS and "avx512f" not in _FLAGS


def has_gpu():
    try:
        import jax
        return jax.default_backend() == "gpu"
    except Exception:  # noqa: BLE001
        return False


def gpu_name():
    """Device kind of jax's first device, as golden/scores/<id>.json records it ("NVIDIA RTX A5000", "cpu", ...)."""
    import jax
    return jax.devices()[0].device_kind


def torch_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def pytest_configure(config):
    for m in ("gpu: needs a GPU", "slow: full test split or 200 training steps", "data: needs YGGDRASIL_DATA",
              "ckpt: needs the checkpoints under YGG_RUNS"):
        config.addinivalue_line("markers", m)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "data" in item.keywords and not DATA.exists():
            item.add_marker(pytest.mark.skip(reason=f"no data at {DATA} (set YGGDRASIL_DATA)"))
        if "ckpt" in item.keywords and not RUNS.exists():
            item.add_marker(pytest.mark.skip(reason=f"no checkpoints at {RUNS} (set YGG_RUNS)"))


# ------------------------------------------------------------------------------------------------------- adapters
# Package layout: yggdrasil.config (Config, parse, upgrade_config), yggdrasil.model (load, forward),
# yggdrasil.train (stored_params, build_task, fit, evaluate), yggdrasil.inference (Yggdrasil.load/init_state/step/
# score), baselines.build, tasks.get. Each adapter states the call it makes. The goldens name config fields as
# yggdrasil.config.Config does.

def make_config(**fields):
    """Config fields (dict) -> yggdrasil.config.Config."""
    from yggdrasil.config import Config
    return Config(**fields)


def flags_to_fields(flags):
    """["--task", "enwik8", "--d_state", "64", ...] -> {"task": "enwik8", "d_state": 64, ...} typed by Config."""
    import dataclasses
    from yggdrasil.config import Config
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for k, v in zip(flags[::2], flags[1::2]):
        k = k.lstrip("-")
        t = types.get(k, str)
        t = {"int": int, "float": float, "str": str, "bool": bool}.get(t, t) if isinstance(t, str) else t
        out[k] = (str(v).lower() in ("1", "true", "yes")) if t is bool else t(v)
    return out


def checkpoint(cid):
    """Path of the reported checkpoint cid (an id of golden/checkpoints.json): $YGG_RUNS/<id>.npz."""
    return RUNS / f"{cid}.npz"


def entry_config(entry):
    """Config of one record of golden/checkpoints.json."""
    return make_config(**{k: v for k, v in entry["config"].items() if v is not None and k in _config_fields()})


def config_file(cid):
    """The config file of a reported checkpoint (golden/checkpoints.json id) -> Config via
    yggdrasil.config.parse(["--config", path])."""
    from yggdrasil.config import parse
    ablation = {"ablation_store_outputs": "ablation/store_outputs", "ablation_token_routing": "ablation/token_routing"}
    task, _, size = cid.partition("_")
    path = ablation.get(cid) or (f"{task}/{size}" if task in ("enwik8", "hg38") else f"state/{cid}")
    path = f"configs/{path}.yaml"
    return parse(["--config", str(REPO / path)])


def run_record_config(cid):
    """Config of the run record $YGG_RUNS/<id>.json next to the checkpoint, read by yggdrasil.config.upgrade_config."""
    from yggdrasil.config import upgrade_config
    rec = json.loads(checkpoint(cid).with_suffix(".json").read_text())
    return entry_config({"config": upgrade_config(rec["config"])})


def load_params(npz):
    """A reported checkpoint (.npz, per-expert arrays) -> params for yggdrasil.model.forward (yggdrasil.model.load)."""
    from yggdrasil import model
    return model.load(npz)


def forward_logp(params, x, cfg):
    """Hard-routing log-probabilities (B, T, classes) of tokens x (B, T): yggdrasil.model.forward(params, x, cfg)."""
    from yggdrasil import model
    return np.asarray(model.forward(params, np.asarray(x, np.int32), cfg, mode="hard")[0])


def stored_count(cfg, vocab, classes):
    """Stored parameters of a fresh model at cfg, routed experts counted per expert as checkpoints store them."""
    from yggdrasil import train
    return int(train.stored_params(cfg, vocab, classes))


def ygg_load(npz):
    """yggdrasil.inference.Yggdrasil.load(path): the streaming model."""
    from yggdrasil.inference import Yggdrasil
    return Yggdrasil.load(npz)


def ygg_step_logp(ygg, tokens):
    """Feed tokens (T,) one at a time through ygg.step (one stream, from ygg.init_state(1)) -> log-probabilities
    (T, classes) of the next token. Works for any backend whose step returns (logits (1, V), state)."""
    state, out = ygg.init_state(1), []
    for t in tokens:
        logits, state = ygg.step(np.array([t], np.int32), state)
        out.append(_numpy(logits)[0])
    z = np.asarray(out, np.float64)
    return z - (np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) + z.max(-1, keepdims=True))


def _numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def triton_available():
    """Hook of the Triton GPU backend tests: a CUDA device, torch and triton, and jax (yggdrasil.inference reads the
    checkpoint with it), all in one environment."""
    try:
        import jax  # noqa: F401
        import torch
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def ygg_load_triton(npz):
    """The Triton GPU backend: yggdrasil.inference.Yggdrasil.load(path, backend="triton"), same step/score interface."""
    from yggdrasil.inference import Yggdrasil
    return Yggdrasil.load(npz, backend="triton")


def ygg_score(ygg, tokens):
    """Yggdrasil.score on one stream tokens (n + 1,) -> bits per token of tokens[1:]."""
    tokens = np.asarray(tokens, np.int32)
    return float(_numpy(ygg.score(tokens[None])).sum() / (len(tokens) - 1))


def eval_test_score(npz, entry):
    """Score on the full test split, evaluated as at the end of training: bits per token (text, windows of seq_len)
    or the dict {"acc", "scaled", ...} (state tasks). entry: one record of golden/checkpoints.json."""
    from yggdrasil import train
    return train.evaluate(entry_config(entry), load_params(npz), "test")


def _config_fields():
    import dataclasses
    from yggdrasil.config import Config
    return {f.name for f in dataclasses.fields(Config)}


def train_trace(cfg, steps):
    """Losses of steps 1..steps (the loss before each step's update) of yggdrasil.train.fit at cfg."""
    import dataclasses
    from yggdrasil import train
    cfg = dataclasses.replace(cfg, steps=steps)
    return [float(loss) for _, _, _, loss, _ in train.fit(cfg, train.build_task(cfg))]


def first_step(cfg):
    """(loss, global gradient norm before clipping) of the first training step of yggdrasil.train.fit at cfg."""
    from yggdrasil import train
    _, _, _, loss, grad_norm = next(iter(train.fit(cfg, train.build_task(cfg))))
    return float(loss), float(grad_norm)


# golden/baseline_logits.json "fields" -> baselines.build config keys. "attn" (the attention kernel) has no
# config key; DeltaProduct's Householder count is in its model name (deltaproduct2, deltaproduct4).
BASELINE_FIELDS = {"d": "d_model", "layers": "n_layers", "state": "d_state", "head_dim": "head_dim"}
BASELINE_DROPPED = ("attn", "householder")


def baseline_logits(name, task, fields, x, device, seed):
    """baselines.build({"model": name, "task": task, **fields}) after torch.manual_seed(seed), run on tokens x (B, T)
    under autocast to the module's AMP type -> (float32 logits (B, T, classes), parameter count)."""
    import torch
    import baselines
    fields = {BASELINE_FIELDS.get(k, k): v for k, v in fields.items() if k not in BASELINE_DROPPED}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    model = baselines.build({"model": name, "task": task, **fields}).to(device).float().eval()
    amp = getattr(baselines.module(name), "AMP", None)
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None):
        lg = model(torch.as_tensor(np.asarray(x, np.int64), device=device)).float().cpu().numpy()
    return lg, sum(p.numel() for p in model.parameters())


def state_task(name, seed):
    """tasks.get(name, seed): .sets[split] = (x, y, last, lengths), .batch(rng, size) -> (x, y, mask)."""
    import tasks
    return tasks.get(name, seed=seed)


def lm_task(name, seq_len, val_windows, test_tokens):
    """tasks.get(name, seq_len=, val_windows=, test_tokens=): .sets[split] = (x, y) windows, .batch(rng, size)."""
    import tasks
    return tasks.get(name, seq_len=seq_len, val_windows=val_windows, test_tokens=test_tokens)


# ------------------------------------------------------------------------------------------------------ comparisons

def logp_close(got, want, rtol=1e-5, atol=1e-5, other_cpu_atol=2e-4):
    """Same kind of CPU as the goldens: |got - want| <= atol + rtol |want| on log-probabilities. Another CPU: the
    probabilities within other_cpu_atol (on an AVX-512 Xeon the 256-token A5 and modular-arithmetic models move them
    by up to 8e-5; log-probabilities near e^-300 differ there by ~1e-3 without meaning)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    if SAME_CPU:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(np.exp(got), np.exp(want), rtol=0, atol=max(atol, other_cpu_atol))


def digest(*arrays):
    """sha256 over dtype, shape and bytes of each array, as golden/tasks.json holds them."""
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()
