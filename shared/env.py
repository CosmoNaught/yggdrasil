"""Python environments of the training and timing scripts. use(name) re-executes the calling script with that
environment's interpreter when it runs under another one, so `python models/<model>/training/train.py ...` works
from any Python.

Environments live in <repo>/.venvs/<name> (or $YGG_VENVS/<name>) and are built by shared/setup_envs.sh from
shared/requirements/<name>.txt:

  jax         Yggdrasil (models/yggdrasil)
  torch       every baseline except the Transformer
  flashattn   the Transformer (flash-attn has no wheel for the torch of the torch environment)
  infer       int8 export and checks (models/yggdrasil/inference)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("jax", "torch", "flashattn", "infer")


def venv(name):
    if name not in NAMES:
        raise KeyError(f"unknown environment {name}; expected one of {NAMES}")
    return Path(os.environ.get("YGG_VENVS", ROOT / ".venvs")) / name


def python(name):
    return str(venv(name) / "bin" / "python")


def use(name):
    path = venv(name)
    if Path(sys.prefix).resolve() != path.resolve():
        if not path.exists():
            sys.exit(f"environment {path} not found; build it with: bash shared/setup_envs.sh {name}")
        os.execv(python(name), [python(name), os.path.abspath(sys.argv[0]), *sys.argv[1:]])
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
