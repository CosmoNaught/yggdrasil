"""Per-model Python environments. use(name) re-executes the script under <repo>/.venvs/<name> (or $YGG_VENVS/<name>).

Built by shared/setup_envs.sh: jax (Yggdrasil), torch (baselines), flashattn (Transformer), infer (int8 export).
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
