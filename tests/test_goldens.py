"""The goldens agree with the paper's numbers (no model code runs here; these pass as long as tests/golden/ is
intact)."""
import json

import pytest

from conftest import GOLDEN, golden

CKPT = golden("checkpoints.json")


@pytest.mark.parametrize("cid", list(CKPT))
def test_recomputed_score_matches_paper(cid):
    """Golden test score (golden/scores) = the score stored at training time to 1e-5 = the paper to 3 d.p."""
    e = CKPT[cid]
    path = GOLDEN / "scores" / f"{cid}.json"
    if not path.exists():
        pytest.skip(f"{path} not recorded")
    s = json.loads(path.read_text())["test"]
    s = s["acc"] if isinstance(s, dict) else s
    assert round(s, 3) == e["paper"], (s, e["paper"])
    assert abs(s - e["json_test"]) < 1e-5, (s, e["json_test"])


@pytest.mark.parametrize("cid", list(CKPT))
def test_seed_summary_matches_paper(cid):
    e = CKPT[cid]
    assert e["paper_match"] and e["paper_mean_match"]
    if e["paper_stored"]:
        assert abs(e["stored_params"] - e["paper_stored"]) < 0.05e6
