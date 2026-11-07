"""yggdrasil.model.forward and yggdrasil.inference on the reported checkpoints: fixed-batch log-probabilities and the
streamed score equal golden/logits/<id>.npz (CPU)."""
import numpy as np
import pytest

from conftest import DATA, GOLDEN, SAME_CPU, checkpoint, entry_config, forward_logp, golden, load_params, logp_close, \
    ygg_load, ygg_score, ygg_step_logp

CKPT = golden("checkpoints.json")
pytestmark = pytest.mark.ckpt


@pytest.fixture(scope="module", params=list(CKPT))
def case(request):
    cid = request.param
    return cid, CKPT[cid], np.load(GOLDEN / "logits" / f"{cid}.npz")


def test_forward_logp(case):
    cid, e, g = case
    logp_close(forward_logp(load_params(checkpoint(cid)), g["x"], entry_config(e)), g["logp"])


def test_step_matches_forward(case):
    """Token-by-token Yggdrasil.step on row 0 of the batch gives the parallel forward's log-probabilities. A
    sequential recurrence sums in another order than the parallel scan: the two differ by up to 8.4e-5 after 256
    tokens (S5, A5, modular arithmetic), hence 2e-4 here instead of 1e-5."""
    cid, e, g = case
    ygg = ygg_load(checkpoint(cid))
    logp_close(ygg_step_logp(ygg, g["x"][0]), g["logp"][0], rtol=2e-4, atol=2e-4)


@pytest.mark.data
def test_stream_score(case):
    """Yggdrasil.score over the first 2048 test tokens, state carried = stream_bits (scored in chunks of 512)."""
    cid, e, g = case
    if "stream_bits" not in g.files:
        pytest.skip("state task: no stream")
    task = e["config"]["task"]
    ids = np.fromfile(DATA / "enwik8", np.uint8)[95_000_000:] if task == "enwik8" else \
        np.fromfile(DATA / task / "test.u8", np.uint8, count=int(g["stream_tokens"]) + 1)
    bits = ygg_score(ygg_load(checkpoint(cid)), ids[:int(g["stream_tokens"]) + 1])
    np.testing.assert_allclose(bits, float(g["stream_bits"]), rtol=1e-6 if SAME_CPU else 1e-5)
