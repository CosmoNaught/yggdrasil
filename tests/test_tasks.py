"""tasks.get: evaluation sets and training batches are bit-identical to golden/tasks.json (sha256)."""
import numpy as np
import pytest

from conftest import digest, golden, lm_task, state_task

G = golden("tasks.json")


@pytest.mark.parametrize("key", list(G["state"]))
def test_state_sets(key):
    name, seed = key.split("/")
    g, t = G["state"][key], state_task(name, int(seed))
    assert (t.vocab, t.classes) == (g["vocab"], g["classes"])
    assert digest(*(np.asarray(a, np.int64) for a in t.sets["test"])) == g["test"]
    assert digest(*(np.asarray(a, np.int64) for a in t.sets["val"])) == g["val"]
    x, y, m = t.batch(np.random.default_rng(int(seed)), 256)
    assert digest(np.asarray(x, np.int64), np.asarray(y, np.int64), np.asarray(m, np.float32)) == g["train_batch"]


@pytest.mark.data
@pytest.mark.parametrize("key", list(G["lm"]))
def test_lm_sets(key):
    name, seed = key.split("/")
    g = G["lm"][key]
    test_tokens = {"enwik8": 5_000_000, "hg38": 253_886_464}[name]
    t = lm_task(name, g["seq_len"], 512, test_tokens)
    assert t.vocab == g["vocab"]
    x, y, m = t.batch(np.random.default_rng(int(seed)), g["batch"])
    assert digest(np.asarray(x, np.int64), np.asarray(y, np.int64), np.asarray(m, np.float32)) == g["train_batch"]
    if "test" in g:
        assert digest(*(np.asarray(a, np.int64) for a in t.sets["test"])) == g["test"]
        assert digest(*(np.asarray(a, np.int64) for a in t.sets["val"])) == g["val"]
