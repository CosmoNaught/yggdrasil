"""PD-SSM (Terzic et al., NeurIPS 2025) from IBM/expressive-sparse-state-space-model (Apache-2.0), vendored in third_party/."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
if __name__ == "__main__":
    import env
    env.use("torch")

SRC = (Path(__file__).resolve().parent / "third_party" / "expressive-sparse-state-space-model" / "state_tracking_PyTorch"
       / "models" / "pdssm.py")


def add_args(ap):
    ap.add_argument("--d", type=int, default=128, help="embedding size")
    ap.add_argument("--state", type=int, default=None, help="state size (default --d)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dictionary", type=int, default=8)


def _module():
    if "pdssm_authors" not in sys.modules:
        spec = importlib.util.spec_from_file_location("pdssm_authors", SRC)
        sys.modules["pdssm_authors"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["pdssm_authors"])
    return sys.modules["pdssm_authors"]


def build(cfg, vocab, classes):
    return _module().PD(output_size=classes, input_size=vocab, state_size=getattr(cfg, "state", None) or cfg.d,
                        embed_size=cfg.d, dictionary_size=getattr(cfg, "dictionary", 8), return_all_outputs=True,
                        num_layers=cfg.layers)


if __name__ == "__main__":
    import trainer
    trainer.main("pdssm", build, add_args)
