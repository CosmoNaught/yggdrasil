"""Full-test-split scores of every reported checkpoint equal golden/scores/<id>.json.

Text scores: each golden names its GPU ("device"; fp32 matmuls). On a GPU of the same type the score must agree to
1e-6 bits. Another GPU type sums in another order (on an A100 the large enwik8 model moves by 1.1e-6 bits), so across
GPU types the tolerance is 2e-6. The text tests skip without a GPU (the large hg38 test split is 254M bases);
YGG_SCORE_ON_CPU=1 runs them anyway. State-task accuracies are from the CPU and are exact counts out of 8192.
"""
import json
import os

import numpy as np
import pytest

from conftest import GOLDEN, checkpoint, golden, gpu_name, has_gpu, eval_test_score as score

CKPT = golden("checkpoints.json")
LM = [c for c, e in CKPT.items() if e["kind"] == "lm"]
STATE = [c for c, e in CKPT.items() if e["kind"] == "state"]
pytestmark = [pytest.mark.slow, pytest.mark.ckpt]


def _golden(cid):
    return json.loads((GOLDEN / "scores" / f"{cid}.json").read_text())


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.parametrize("cid", LM)
def test_lm_score(cid):
    if not has_gpu() and not os.environ.get("YGG_SCORE_ON_CPU"):
        pytest.skip("LM scores are recorded on GPU")
    g = _golden(cid)
    s = score(checkpoint(cid), CKPT[cid])
    atol = 1e-6 if gpu_name() == g["device"] else 2e-6
    np.testing.assert_allclose(float(s), g["test"], rtol=0, atol=atol, err_msg=f"on {gpu_name()}, golden on {g['device']}")
    assert round(float(s), 3) == CKPT[cid]["paper"]


@pytest.mark.parametrize("cid", STATE)
def test_state_score(cid):
    g = _golden(cid)["test"]
    s = score(checkpoint(cid), CKPT[cid])
    assert abs(s["acc"] - g["acc"]) < 1e-6, (s, g)
    assert round(s["acc"], 3) == CKPT[cid]["paper"]
