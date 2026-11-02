"""PD-SSM training.

Source: the authors' code for Terzic et al., "Structured Sparse Transition Matrices to Enable State Tracking in
State-Space Models" (NeurIPS 2025), https://github.com/IBM/expressive-sparse-state-space-model at commit
8682e78101be84f67ceb64702855e5d9e820f7d2 (Apache-2.0), vendored unmodified in
third_party/expressive-sparse-state-space-model/ (LICENSE there). Module state_tracking_PyTorch/models/pdssm.py class
PD, built as in their state_tracking_PyTorch/train.py: embedding size = --d, state size = --state (default --d, as in
their script), return_all_outputs=True, --layers blocks, their defaults otherwise (dictionary 8, dropout 0.01).
PD already holds a standard token embedding (nn.Embedding) and a linear readout around the PD blocks, so the same
untouched class serves the language-model tasks (enwik8, hg38) and the state tasks. Their recurrence is a plain PyTorch
loop over time; no custom kernels.

    python models/pdssm/training/train.py --task <task> --d <width> --layers <layers> [--state <state>] --lr <lr> --seed <seed>

"""
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
