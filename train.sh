#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MICROMAMBA="$ROOT/.tools/micromamba/bin/micromamba"
MAMBA_ROOT_PREFIX="$ROOT/.micromamba"

export MAMBA_ROOT_PREFIX MAMBA_NO_BANNER=1
export FACEOCC_SEED="${FACEOCC_SEED:-42}"
export PYTHONHASHSEED="$FACEOCC_SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

[[ -x "$MICROMAMBA" ]] || { printf 'Run ./setup.sh first.\n' >&2; exit 1; }
[[ -d "$ROOT/Dataset/CelebA-HQ-align" && -d "$ROOT/Dataset/CelebAMask-HQ-align" ]] || { printf 'Run ./prepare.sh first.\n' >&2; exit 1; }

cd "$ROOT"
exec "$MICROMAMBA" run -n faceocc python train.py --seed "$FACEOCC_SEED" "$@"
