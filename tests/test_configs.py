"""Config files and run records: configs/<...>.yaml (yggdrasil.config.parse) and the run record next to each
checkpoint (yggdrasil.config.upgrade_config) give the configuration of golden/checkpoints.json, field by field."""
import pytest

from conftest import config_file, golden, run_record_config

CKPT = golden("checkpoints.json")


@pytest.mark.parametrize("cid", list(CKPT))
def test_config_file(cid):
    cfg, want = config_file(cid), CKPT[cid]["config"]
    assert {k: getattr(cfg, k) for k in want} == want


@pytest.mark.ckpt
@pytest.mark.parametrize("cid", list(CKPT))
def test_run_record(cid):
    cfg, want = run_record_config(cid), CKPT[cid]["config"]
    assert {k: getattr(cfg, k) for k in want} == want
