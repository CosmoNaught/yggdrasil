"""Run models/<model>/training/train.py in that model's environment.

    python shared/train.py --model <model> --task <task> [flags]
"""
import os
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
ALIAS = {"deltaproduct2": ("deltaproduct", ["--householder", "2"]),
         "deltaproduct4": ("deltaproduct", ["--householder", "4"])}


def split_model(argv):
    rest, model, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--model":
            model, i = argv[i + 1], i + 2
        elif argv[i].startswith("--model="):
            model, i = argv[i].split("=", 1)[1], i + 1
        else:
            rest.append(argv[i])
            i += 1
    return model, rest


def main():
    model, rest = split_model(sys.argv[1:])
    if model is None:
        sys.exit(__doc__)
    model, extra = ALIAS.get(model, (model, []))
    prog = MODELS / model / "training" / "train.py"
    if not prog.exists():
        sys.exit(f"unknown model {model}")
    os.execv(sys.executable, [sys.executable, str(prog), *extra, *rest])


if __name__ == "__main__":
    main()
