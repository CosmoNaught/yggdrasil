"""baselines.build: logits on a fixed batch at a fixed init seed and the parameter count of every baseline equal
golden/baseline_logits.npz. "/cpu" goldens are from the CPU, "/cuda" goldens from an NVIDIA RTX A5000 (Mamba-2, Gated
DeltaNet and DeltaProduct need the GPU). The models run under autocast to their AMP type (bf16 for the Transformer
and DeltaProduct). On the golden's GPU every model agrees to 1e-5; on another GPU (an A100) the bf16 models differ by
one bf16 ulp in under 0.3% of the logits, so there they are compared at bf16 resolution."""
import numpy as np
import pytest

from conftest import GOLDEN, baseline_logits, golden, torch_gpu

pytest.importorskip("torch")  # the baselines run in the torch environment

META = golden("baseline_logits.json")


def _case(key):
    return pytest.param(key, marks=[pytest.mark.gpu] if key.endswith("/cuda") else [])


@pytest.mark.parametrize("key", [_case(k) for k in META])
def test_baseline_logits(key):
    name, device = key.split("/")
    if device == "cuda" and not torch_gpu():
        pytest.skip("recorded on GPU")
    g, m = np.load(GOLDEN / "baseline_logits.npz"), META[key]
    lg, n_params = baseline_logits(name, m["task"], m["fields"], g["x"], device, m["seed"])
    assert n_params == m["n_params"]
    tol = 1e-5
    if device == "cuda" and m["amp"] != "None":
        import torch
        if torch.cuda.get_device_name() != m["device"]:
            tol = 4e-3  # 2 ulp of bf16 at |logit| ~ 0.5
            assert np.abs(lg - g[key]).mean() < 1e-4
    np.testing.assert_allclose(lg, g[key], rtol=tol, atol=tol)
