"""The fused Triton step (yggdrasil/inference_triton.py) against the JAX step on the paper checkpoints. Needs a GPU,
torch, triton and jax in one environment.

    python -m unittest cpu/tests/test_triton.py

YGG_RUNS and YGGDRASIL_DATA as in test_checkpoints.py.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the reference runs on the CPU
sys.path.insert(0, str(REPO))
from tasks.text import DATA, streams  # noqa: E402

try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False

RUNS = Path(os.environ.get("YGG_RUNS", REPO / "results"))
CHECKPOINTS = {"enwik8": RUNS / "enwik8_small.npz", "hg38": RUNS / "hg38_medium.npz"}
B, S = 64, 256  # streams, steps
ATOL = 1e-4     # log-probabilities, fp32
FLIPS = 1e-3    # fraction of stream steps allowed beyond ATOL (routing ties)


def stream_tokens(task):
    """(B, S) tokens: stream b is a contiguous stretch of the test split."""
    return np.asarray(streams(task)[2][:B * S], np.int64).reshape(B, S)


@unittest.skipUnless(GPU, "no GPU")
class TritonMatchesJax(unittest.TestCase):
    def test_step(self):
        import jax
        from yggdrasil.inference import Yggdrasil
        for task, path in CHECKPOINTS.items():
            with self.subTest(task=task):
                if not Path(path).exists() or not DATA.exists():
                    self.skipTest(f"no {path} or no data in {DATA}")
                x = stream_tokens(task)
                ref, fused, graph = Yggdrasil.load(path), Yggdrasil.load(path, backend="triton"), \
                    Yggdrasil.load(path, backend="triton", cuda_graph=True)
                # every step starts from the reference state, so a rare routing flip at a node score within float
                # rounding of 0 changes one step of one stream instead of the rest of its sequence
                state, worst, off = ref.init_state(B), 0.0, 0
                for t in range(S):
                    h = torch.tensor(np.asarray(state).transpose(1, 0, 2), device="cuda")
                    l1, _ = fused.step(x[:, t], h)
                    l2, _ = graph.step(x[:, t], h)
                    l0, state = ref.step(x[:, t], state)
                    lp0 = np.asarray(jax.nn.log_softmax(l0, -1))
                    for lp in (torch.log_softmax(l1, -1), torch.log_softmax(l2, -1)):
                        d = np.abs(lp.cpu().numpy() - lp0).max(-1)
                        off += int((d > ATOL).sum())
                        worst = max(worst, float(d[d <= ATOL].max(initial=0.0)))
                print(f"\n{task}: {B} streams x {S} steps, {off} of {2 * B * S} stream steps beyond {ATOL}, "
                      f"largest difference of the others {worst:.1e}")
                self.assertLessEqual(off, FLIPS * 2 * B * S)
                w = x[:8, :129]
                np.testing.assert_allclose(fused.score(w), ref.score(w), rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
