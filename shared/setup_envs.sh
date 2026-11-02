#!/bin/bash
# Builds the Python environments of shared/env.py with uv from shared/requirements/<name>.txt.
#   bash shared/setup_envs.sh                  all four: jax torch flashattn infer
#   bash shared/setup_envs.sh torch infer      only these
#   YGG_VENVS=<dir> bash shared/setup_envs.sh  build under <dir> instead of <repo>/.venvs
# torch.txt and flashattn.txt pull CUDA 12.8 wheels (--extra-index-url .../whl/cu128); infer.txt pulls the CPU build
# (.../whl/cpu). Edit those lines for another CUDA version. mamba-ssm and causal-conv1d build against the installed torch.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENVS=${YGG_VENVS:-$ROOT/.venvs}
REQ=$ROOT/shared/requirements
for name in ${@:-jax torch flashattn infer}; do
  venv=$VENVS/$name
  echo "== $name -> $venv"
  uv venv -p 3.12 "$venv"
  pip=(uv pip install -p "$venv/bin/python" --index-strategy unsafe-best-match)
  if [ "$name" = torch ]; then
    "${pip[@]}" -r <(grep -E '^(--|torch==|triton==|numpy==)' "$REQ/torch.txt") setuptools wheel ninja packaging
    "${pip[@]}" --no-build-isolation -r "$REQ/torch.txt"
  else
    "${pip[@]}" -r "$REQ/$name.txt"
  fi
done
