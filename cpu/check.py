"""Bits per token of the int8 kernel against the float model (yggdrasil/inference.py) on the first test windows.

    python cpu/check.py --npz results/<run>.npz --model model.bin [--windows 32] [--int_ref]

--int_ref also runs export.int_forward and reports its largest logit difference to the kernel (0: bit-exact).
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
sys.path.insert(0, str(HERE))
import export as E  # noqa: E402

TOLERANCE = 0.02  # bits per token, kernel against the float model


def kernel_logits(ygg, model, x):
    """ygg logits of windows x (S, T) -> (S, T, V)."""
    with tempfile.TemporaryDirectory() as tmp:
        xf, of = os.path.join(tmp, "x.i32"), os.path.join(tmp, "logits.f32")
        x.astype(np.int32).tofile(xf)
        subprocess.run([ygg, "logits", model, xf, str(len(x)), str(x.shape[1]), of], check=True,
                       stdout=subprocess.DEVNULL)
        return np.fromfile(of, np.float32).reshape(*x.shape, -1)


def check(npz, model, ygg=str(HERE / "ygg"), windows=32, task="", seq_len=0, data=E.DATA, int_ref=False):
    """Float and int8 bits per token on the first test windows, and with int_ref the kernel against int_forward."""
    task, seq_len = E.checkpoint_task(npz, task, seq_len)
    x, y = E.windows(E.streams(task, data)[2], seq_len, windows)
    res = dict(npz=str(npz), model=str(model), windows=len(x), seq_len=seq_len)
    res["bpc_float"] = float(E.inference.Yggdrasil.load(npz).score(np.concatenate([x, y[:, -1:]], 1)).sum() / y.size)
    li = kernel_logits(ygg, model, x)
    res["bpc_int8"] = E.bpc(li, y)
    res["diff"] = res["bpc_int8"] - res["bpc_float"]
    if int_ref:
        ex = json.loads(Path(str(model) + ".json").read_text())
        _, layers, output_bias = E.hard_layers(npz)
        q = E.quantize(layers, [E.LayerStats(**s) for s in ex["stats"]], **ex["quantize"])
        res["max_abs_logit_kernel_vs_numpy"] = float(np.abs(E.int_forward(q, output_bias, x) - li).max())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ygg", default=str(HERE / "ygg"))
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--task", default="")
    ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--data", default=str(E.DATA))
    ap.add_argument("--int_ref", action="store_true")
    a = ap.parse_args()
    res = check(a.npz, a.model, a.ygg, a.windows, a.task, a.seq_len, a.data, a.int_ref)
    print(json.dumps(res))
    print(f"float {res['bpc_float']:.4f}  int8 {res['bpc_int8']:.4f}  diff {res['diff']:+.4f}  (tolerance {TOLERANCE})")


if __name__ == "__main__":
    main()
