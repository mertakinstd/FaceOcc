#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MICROMAMBA="$ROOT/.tools/micromamba/bin/micromamba"
MAMBA_ROOT_PREFIX="$ROOT/.micromamba"

export MAMBA_ROOT_PREFIX MAMBA_NO_BANNER=1

[[ "$#" -eq 0 ]] || { printf 'train.sh takes no arguments.\n' >&2; exit 1; }
[[ -x "$MICROMAMBA" ]] || { printf 'Run ./setup.sh first.\n' >&2; exit 1; }
[[ -d "$ROOT/Dataset/CelebA-HQ-align" && -d "$ROOT/Dataset/CelebAMask-HQ-align" ]] || { printf 'Run ./prepare.sh first.\n' >&2; exit 1; }

cd "$ROOT"
exec "$MICROMAMBA" run -n faceocc python train.py
