"""Compare ygg with export.int_forward on random models (default, --blocked, --prefetch). 0 = bit-exact.

    python test_kernel.py [--ygg ./ygg]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export as E  # noqa: E402

# n, b, D, V, layers, out_rank, lower_rank, shared_out
CONFIGS = [(128, 16, 8, 256, 2, 8, 16, 0), (128, 8, 6, 256, 2, 16, 32, 0), (64, 16, 5, 27, 3, 32, 20, 1),
           (64, 8, 9, 5, 2, 8, 0, 0), (128, 16, 7, 5, 3, 16, 4, 1), (128, 16, 4, 256, 1, 8, 0, 0),
           (256, 16, 6, 256, 3, 32, 64, 0), (256, 16, 5, 5, 5, 32, 64, 0), (256, 8, 5, 256, 2, 16, 48, 1),
           (256, 16, 4, 27, 2, 8, 20, 0)]
GROUP = [(256, 16, 6, 256, 3, 32, 64, 0), (128, 8, 6, 256, 3, 16, 32, 1), (64, 16, 5, 5, 4, 8, 0, 0)]  # group scales
FACTORED = [((128, 16, 7, 256, 3, 16, 32, 0), 32), ((256, 16, 5, 256, 3, 32, 64, 0), 64), ((64, 8, 6, 5, 2, 8, 0, 1), 16),
            ((128, 16, 6, 27, 2, 16, 16, 0), 4)]  # factored storage (config, rank)

ap = argparse.ArgumentParser()
ap.add_argument("--ygg", default=str(HERE / "ygg"))
a = ap.parse_args()
worst = 0.0
with tempfile.TemporaryDirectory() as tmp:
    mb, xf, of = (os.path.join(tmp, f) for f in ("m.bin", "x.i32", "o.f32"))
    for c, grp, fr in [(c, False, 0) for c in CONFIGS] + [(c, True, 0) for c in GROUP] + \
            [(c, i % 2 == 0, r) for i, (c, r) in enumerate(FACTORED)]:
        n, b, D, V, L, R, R2, sh = c
        Qs = E.random_quantized(n, b, D, V, L, R, R2, sh, seed=7, group=grp, factored=fr)
        hb = np.linspace(-1, 1, V)
        E.write(Qs, hb, mb, {})
        x = np.random.default_rng(0).integers(0, V, (8, 64)).astype(np.int32)
        x.tofile(xf)
        ref = E.int_forward(Qs, hb, x)
        diffs = []
        for fl in ([], ["--blocked"], ["--prefetch"]):
            subprocess.run([a.ygg, "logits", mb, xf, "8", "64", of, *fl], check=True, stdout=subprocess.DEVNULL)
            diffs.append(float(np.abs(np.fromfile(of, np.float32).reshape(ref.shape) - ref).max()))
        worst = max(worst, *diffs)
        print("n,b,D,V,L,R,R2,shared", c, "group scales" if grp else "", f"factored rank {fr}" if fr else "", "max |kernel - numpy| default / blocked / prefetch", diffs, flush=True)
print("PASS (bit-exact)" if worst == 0 else f"FAIL max diff {worst}")
