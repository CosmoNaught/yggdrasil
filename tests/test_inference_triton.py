"""The Triton GPU backend of yggdrasil.inference on the reported checkpoints: token-by-token log-probabilities equal
golden/logits/<id>.npz (CPU, parallel forward) and the streamed score equals its stream_bits. Skipped without a CUDA
device and jax, torch and triton in one environment (conftest.triton_available). Tolerances as for the sequential CPU
step (test_logits.py): the step sums in another order than the parallel scan."""
import numpy as np
import pytest

from conftest import DATA, GOLDEN, checkpoint, golden, triton_available, ygg_load_triton, ygg_score, ygg_step_logp

CKPT = golden("checkpoints.json")
pytestmark = [pytest.mark.gpu, pytest.mark.ckpt,
              pytest.mark.skipif(not triton_available(), reason="needs a CUDA device and jax, torch and triton in one environment")]


@pytest.mark.parametrize("cid", list(CKPT))
def test_triton_step(cid):
    g = np.load(GOLDEN / "logits" / f"{cid}.npz")
    logp = ygg_step_logp(ygg_load_triton(checkpoint(cid)), g["x"][0])
    np.testing.assert_allclose(logp, g["logp"][0], rtol=2e-4, atol=2e-4)


@pytest.mark.data
@pytest.mark.parametrize("cid", [c for c, e in CKPT.items() if e["kind"] == "lm"])
def test_triton_score(cid):
    g = np.load(GOLDEN / "logits" / f"{cid}.npz")
    n, task = int(g["stream_tokens"]), CKPT[cid]["config"]["task"]
    ids = np.fromfile(DATA / "enwik8", np.uint8)[95_000_000:] if task == "enwik8" else \
        np.fromfile(DATA / task / "test.u8", np.uint8, count=n + 1)
    bits = ygg_score(ygg_load_triton(checkpoint(cid)), ids[:n + 1])
    np.testing.assert_allclose(bits, float(g["stream_bits"]), rtol=1e-5)
