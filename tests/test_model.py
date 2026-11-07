"""yggdrasil.train.stored_params: a fresh model at every reported configuration stores exactly as many parameters as
its checkpoint (golden/checkpoints.json "stored_params", counted from the .npz)."""
import pytest

from conftest import entry_config, golden, stored_count

CKPT = golden("checkpoints.json")
TASKS = golden("tasks.json")


def _sizes(task):
    if task in ("enwik8", "hg38"):
        return {"enwik8": (256, 256), "hg38": (5, 5)}[task]
    g = next(v for k, v in TASKS["state"].items() if k.startswith(task + "/"))
    return g["vocab"], g["classes"]


@pytest.mark.parametrize("cid", list(CKPT))
def test_stored_params(cid):
    e = CKPT[cid]
    assert stored_count(entry_config(e), *_sizes(e["config"]["task"])) == e["stored_params"]
