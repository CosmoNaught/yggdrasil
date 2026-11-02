"""Compare the float model (hard routing) with the int8 kernel on the test windows.

    python check.py --npz results/<run>.npz --model model.bin [--ygg ./ygg] [--test_chars N] [--int_ref]

--int_ref also runs export.int_forward and reports its max logit difference to the kernel.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
sys.path.insert(0, str(HERE))
import export as E  # noqa: E402

LN2 = np.log(2.0)


def float_bpc(P, x, y):
    """Bits per token under train.forward(mode="hard")."""
    import jax
    import jax.numpy as jnp
    import train as main
    params = jax.tree_util.tree_map(jnp.asarray, {"root": P["root"], "branches": P["branches"], "head_b": P["head_b"]})
    nll = jax.jit(lambda p, xx, yy: -jnp.take_along_axis(main.forward(p, xx, mode="hard"), yy[..., None], -1).sum())
    return sum(float(nll(params, x[i:i + 64], y[i:i + 64])) for i in range(0, len(x), 64)) / y.size / LN2


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ygg", default=str(HERE / "ygg"))
    ap.add_argument("--test_chars", type=int, default=0, help="0: the checkpoint's test_chars")
    ap.add_argument("--int_ref", action="store_true")
    a = ap.parse_args()
    meta = json.loads(Path(a.npz).with_suffix(".json").read_text())
    cfg = meta["config"]
    T = cfg["T"]
    x, y = E.windows(E.load_split(cfg["task"], "test"), T, (a.test_chars or cfg["test_chars"]) // T)
    P = E.load_params(a.npz)
    res = dict(npz=a.npz, model=a.model, windows=len(x), T=T, tokens=int(y.size))
    res["bpc_float"] = float_bpc(P, x, y)
    with tempfile.TemporaryDirectory() as tmp:
        xf, of = os.path.join(tmp, "x.i32"), os.path.join(tmp, "logits.f32")
        x.astype(np.int32).tofile(xf)
        subprocess.run([a.ygg, "logits", a.model, xf, str(len(x)), str(T), of], check=True, stdout=subprocess.DEVNULL)
        V = len(P["head_b"])
        li = np.fromfile(of, np.float32).reshape(len(x), T, V)
    res["bpc_int8"] = E.bpc(li, y)
    res["diff"] = res["bpc_int8"] - res["bpc_float"]
    if a.int_ref:
        ex = json.loads(Path(a.model + ".json").read_text())
        Qs = E.quantize(E.hard_layers(P), ex["stats"], ex["hmargin"], ex.get("state_scale") == "group")
        lr = np.concatenate([E.int_forward(Qs, P["head_b"], x[i:i + 64]) for i in range(0, len(x), 64)])
        res["bpc_int8_numpy"] = E.bpc(lr, y)
        res["max_abs_logit_kernel_vs_numpy"] = float(np.abs(lr - li).max())
    if "test" in meta:
        res["main_py_test_bpc"] = meta["test"]
    print(json.dumps(res))
    print(f"float bpc {res['bpc_float']:.4f}  int8 bpc {res['bpc_int8']:.4f}  diff {res['diff']:+.4f}")


if __name__ == "__main__":
    main_()
