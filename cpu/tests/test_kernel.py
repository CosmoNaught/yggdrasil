"""ygg.cpp against export.int_forward on random int8 models: bit-exact.

    YGG=cpu/ygg python -m unittest cpu/tests/test_kernel.py
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CPU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPU))
import export as E  # noqa: E402
from export import KernelConfig as K  # noqa: E402

YGG = os.environ.get("YGG", str(CPU / "ygg"))
STREAMS, STEPS = 8, 64

CASES = [  # d_state, block_size, router_depth, vocab, n_layers, readout_rank, layer_input_rank
    K(128, 16, 8, 256, 2, 8, 16), K(128, 8, 6, 256, 2, 16, 32), K(64, 16, 5, 27, 3, 32, 20),
    K(64, 8, 9, 5, 2, 8, 0), K(128, 16, 7, 5, 3, 16, 4), K(128, 16, 4, 256, 1, 8, 0),
    K(256, 16, 6, 256, 3, 32, 64), K(256, 16, 5, 5, 5, 32, 64), K(256, 8, 5, 256, 2, 16, 48),
    K(256, 16, 4, 27, 2, 8, 20),
    # group scales
    K(256, 16, 6, 256, 3, 32, 64, group=True), K(128, 8, 6, 256, 3, 16, 32, group=True),
    K(64, 16, 5, 5, 4, 8, 0, group=True),
    # factored storage
    K(128, 16, 7, 256, 3, 16, 32, input_rank=32, group=True), K(256, 16, 5, 256, 3, 32, 64, input_rank=64),
    K(64, 8, 6, 5, 2, 8, 0, input_rank=16, group=True), K(128, 16, 6, 27, 2, 16, 16, input_rank=4),
]


def build(cfg, path):
    """Write the random model of cfg to path -> (layers, int_forward logits on the test tokens)."""
    layers = E.random_model(cfg, seed=7)
    output_bias = np.linspace(-1, 1, cfg.vocab)
    E.write(layers, output_bias, path)
    return layers, E.int_forward(layers, output_bias, tokens(cfg))


def tokens(cfg):
    return np.random.default_rng(0).integers(0, cfg.vocab, (STREAMS, STEPS)).astype(np.int32)


@unittest.skipUnless(Path(YGG).exists(), f"no kernel at {YGG} (make -C cpu)")
class KernelTest(unittest.TestCase):
    def test_bit_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            mb, xf, of = (os.path.join(tmp, f) for f in ("m.bin", "x.i32", "o.f32"))
            for cfg in CASES:
                with self.subTest(config=cfg):
                    _, ref = build(cfg, mb)
                    tokens(cfg).tofile(xf)
                    subprocess.run([YGG, "logits", mb, xf, str(STREAMS), str(STEPS), of], check=True,
                                   stdout=subprocess.DEVNULL)
                    np.testing.assert_array_equal(np.fromfile(of, np.float32).reshape(ref.shape), ref)


if __name__ == "__main__":
    unittest.main()
