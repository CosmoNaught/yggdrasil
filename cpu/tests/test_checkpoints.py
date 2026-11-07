"""The int8 kernel on the paper checkpoints: within check.TOLERANCE bits per token of yggdrasil.inference on the first
32 test windows, and bit-exact with export.int_forward.

    python -m unittest cpu/tests/test_checkpoints.py

YGG_RUNS        directory of the released checkpoints <id>.npz and <id>.json (default results/)
YGGDRASIL_DATA  data directory (default data/)
YGG             int8 kernel (default cpu/ygg)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cpu"))
import check  # noqa: E402
import export  # noqa: E402
from tasks.text import DATA  # noqa: E402

YGG = os.environ.get("YGG", str(REPO / "cpu/ygg"))
RUNS = Path(os.environ.get("YGG_RUNS", REPO / "results"))
CHECKPOINTS = {"enwik8": RUNS / "enwik8_small.npz", "hg38": RUNS / "hg38_medium.npz"}


@unittest.skipUnless(Path(YGG).exists(), f"no kernel at {YGG} (make -C cpu)")
class KernelTolerance(unittest.TestCase):
    def test_kernel(self):
        for task, path in CHECKPOINTS.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                if not Path(path).exists() or not DATA.exists():
                    self.skipTest(f"no {path} or no data in {DATA}")
                export.export(path, f"{tmp}/m.bin", data=DATA)
                r = check.check(path, f"{tmp}/m.bin", YGG, 32, data=DATA, int_ref=True)
                print(f"\n{task}: float {r['bpc_float']:.4f}  int8 {r['bpc_int8']:.4f}  diff {r['diff']:+.4f}")
                self.assertLessEqual(abs(r["diff"]), check.TOLERANCE)
                self.assertEqual(r["max_abs_logit_kernel_vs_numpy"], 0.0)


if __name__ == "__main__":
    unittest.main()
