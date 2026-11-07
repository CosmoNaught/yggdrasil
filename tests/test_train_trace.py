"""yggdrasil.train.fit: 200-step loss traces at a fixed seed and the first-step loss and gradient norm equal
golden/traces.json (CPU). Configurations: the paper's small enwik8 and hg38 models (batches shrunk for the CPU) and
the parity model, with the paper's schedule over 200 steps."""
import numpy as np
import pytest

from conftest import SAME_CPU, first_step, flags_to_fields, golden, make_config, train_trace

G = golden("traces.json")
# On the goldens' kind of CPU the whole trace must agree to 1e-5. On another CPU (an AVX-512 Xeon) fp32 rounding
# differs from step 1 and training amplifies it past 1e-3 after step 17 (enwik8), 21 (parity) and 32 (hg38); the
# code that wrote the goldens gives the same losses there. On such a CPU only the first OTHER_CPU_STEPS are compared.
RTOL = 1e-5 if SAME_CPU else 1e-3
OTHER_CPU_STEPS = 10


def _cfg(name):
    return make_config(**flags_to_fields(G[name]["flags"]), eval_every=10**9, save=False)


def _marks(name):
    return [pytest.mark.slow] + ([pytest.mark.data] if name != "parity_check" else [])


@pytest.mark.parametrize("name", [pytest.param(n, marks=_marks(n)) for n in G])
def test_first_step(name):
    loss, grad_norm = first_step(_cfg(name))
    g = G[name]["first_step"]
    np.testing.assert_allclose(loss, g["loss"], rtol=1e-6 if SAME_CPU else 1e-5)
    np.testing.assert_allclose(grad_norm, g["grad_norm"], rtol=1e-5 if SAME_CPU else 1e-4)


@pytest.mark.parametrize("name", [pytest.param(n, marks=_marks(n)) for n in G])
def test_loss_trace(name):
    g = G[name]
    trace = train_trace(_cfg(name), g["steps"])
    assert len(trace) == g["steps"]
    n = g["steps"] if SAME_CPU else OTHER_CPU_STEPS
    np.testing.assert_allclose(trace[:n], g["loss"][:n], rtol=RTOL, atol=1e-7)
