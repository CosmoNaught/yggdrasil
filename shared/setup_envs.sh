#!/bin/bash
# Build the environments of shared/env.py with uv.
#   bash shared/setup_envs.sh [jax] [torch] [flashattn] [infer]
# Wheels target CUDA 12.8; edit --extra-index-url in shared/requirements/*.txt for another version.
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
