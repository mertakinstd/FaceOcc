#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="faceocc"
FACEOCC_ARCHIVE_SIZE="538014369"
CELEBAMASK_ARCHIVE_SIZE="3153930546"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu132"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TOOLS="$ROOT/.tools"
MAMBA_ROOT_PREFIX="$ROOT/.micromamba"
MICROMAMBA="$TOOLS/micromamba/bin/micromamba"
DOWNLOADS="$ROOT/.cache/faceocc-downloads"
FACEOCC_DIR="$ROOT/Dataset/FaceOcc"
CELEBAMASK_DIR="$ROOT/data/raw/CelebAMask-HQ"

export MAMBA_ROOT_PREFIX MAMBA_NO_BANNER=1

log() { printf '\n[FaceOcc setup] %s\n' "$*"; }
die() { printf '\n[FaceOcc setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$#" -eq 0 ]] || die "setup.sh takes no arguments."
[[ -f "$ROOT/.env" ]] || die "Missing .env. Copy .env.example to .env and configure it."
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

[[ -n "${FACEOCC_GDRIVE_URL:-}" ]] || die "FACEOCC_GDRIVE_URL is not set."
[[ -n "${CELEBAMASK_GDRIVE_URL:-}" ]] || die "CELEBAMASK_GDRIVE_URL is not set."
[[ "${CELEBAMASK_LICENSE_ACCEPTED:-0}" == "1" ]] || die "Set CELEBAMASK_LICENSE_ACCEPTED=1 after accepting the CelebAMask-HQ license."

for command in curl tar unzip find stat; do
    command -v "$command" >/dev/null || die "Required command not found: $command"
done
[[ "$(uname -s)" == "Linux" ]] || die "Linux is required."

[[ "$(uname -m)" == "x86_64" ]] || die "Linux x86_64 is required by the current CUDA wheel set."
MAMBA_PLATFORM="linux-64"

install_micromamba() {
    [[ -x "$MICROMAMBA" ]] && return
    log "Installing Micromamba"
    mkdir -p "$TOOLS/micromamba"
    curl --fail --location --silent --show-error \
        "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" \
        | tar -xj -C "$TOOLS/micromamba" bin/micromamba
}

install_environment() {
    if [[ -x "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/python" ]]; then
        log "Updating environment"
        "$MICROMAMBA" env update -n "$ENV_NAME" -f "$ROOT/environment.yml" -y
    else
        log "Creating environment"
        "$MICROMAMBA" create -f "$ROOT/environment.yml" -y
    fi

    "$MICROMAMBA" run -n "$ENV_NAME" python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        --index-url "$PYTORCH_INDEX_URL" \
        "torch==2.13.0+cu132" "torchvision==0.28.0+cu132"
    "$MICROMAMBA" run -n "$ENV_NAME" python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        -r "$ROOT/requirements.txt"
}

download_archive() {
    local url="$1" output="$2" size="$3" name="$4"
    mkdir -p "$DOWNLOADS"

    if [[ -f "$output" ]]; then
        [[ "$(stat -c '%s' "$output")" == "$size" ]] || die "$name exists with an unexpected size; remove it and retry."
        unzip -tq "$output" >/dev/null || die "$name failed ZIP integrity validation."
        return
    fi

    log "Downloading $name"
    "$MICROMAMBA" run -n "$ENV_NAME" python -m gdown --fuzzy "$url" -O "$output"
    [[ "$(stat -c '%s' "$output")" == "$size" ]] || die "$name has an unexpected size."
    unzip -tq "$output" >/dev/null || die "$name failed ZIP integrity validation."
}

extract_faceocc() {
    if [[ -d "$FACEOCC_DIR" ]]; then
        [[ -d "$FACEOCC_DIR/CelebAHQ" && -d "$FACEOCC_DIR/COFW_test" ]] || die "Invalid FaceOcc directory: $FACEOCC_DIR"
        return
    fi

    local tmp="$ROOT/.cache/faceocc-extract"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    unzip -q "$DOWNLOADS/FaceOcc.zip" -d "$tmp"
    local src
    src="$(find "$tmp" -type d -name FaceOcc -print -quit)"
    [[ -n "$src" ]] || die "FaceOcc root was not found in FaceOcc.zip."
    mkdir -p "$(dirname "$FACEOCC_DIR")"
    mv "$src" "$FACEOCC_DIR"
    rm -rf "$tmp"
}

extract_celebamask() {
    if [[ -d "$CELEBAMASK_DIR" ]]; then
        [[ -d "$CELEBAMASK_DIR/CelebA-HQ-img" && -d "$CELEBAMASK_DIR/CelebAMask-HQ-mask-anno" ]] || die "Invalid CelebAMask-HQ directory: $CELEBAMASK_DIR"
        return
    fi

    local tmp="$ROOT/.cache/celebamask-extract"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    unzip -q "$DOWNLOADS/CelebAMask-HQ.zip" -d "$tmp"
    local images src
    images="$(find "$tmp" -type d -name CelebA-HQ-img -print -quit)"
    [[ -n "$images" ]] || die "CelebA-HQ-img was not found in CelebAMask-HQ.zip."
    src="$(dirname "$images")"
    [[ -d "$src/CelebAMask-HQ-mask-anno" ]] || die "CelebAMask-HQ-mask-anno was not found."
    mkdir -p "$(dirname "$CELEBAMASK_DIR")"
    mv "$src" "$CELEBAMASK_DIR"
    rm -rf "$tmp"
}

preflight() {
    "$MICROMAMBA" run -n "$ENV_NAME" python - <<'PY'
import sys
import segmentation_models_pytorch as smp
import torch
import torchvision

assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == "2.13.0+cu132"
assert torchvision.__version__ == "0.28.0+cu132"
assert smp.__version__ == "0.5.0"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for FaceOcc training")
PY

    local count
    count="$(find "$CELEBAMASK_DIR/CelebA-HQ-img" -maxdepth 1 -type f | wc -l | tr -d ' ')"
    [[ "$count" -eq 30000 ]] || die "Expected 30,000 CelebA-HQ images, found $count."
}

install_micromamba
install_environment

if [[ -e "$FACEOCC_DIR" && ( ! -d "$FACEOCC_DIR/CelebAHQ" || ! -d "$FACEOCC_DIR/COFW_test" ) ]]; then
    die "Invalid FaceOcc directory: $FACEOCC_DIR"
fi
if [[ -e "$CELEBAMASK_DIR" && ( ! -d "$CELEBAMASK_DIR/CelebA-HQ-img" || ! -d "$CELEBAMASK_DIR/CelebAMask-HQ-mask-anno" ) ]]; then
    die "Invalid CelebAMask-HQ directory: $CELEBAMASK_DIR"
fi

if [[ ! -d "$FACEOCC_DIR/CelebAHQ" || ! -d "$FACEOCC_DIR/COFW_test" ]]; then
    download_archive "$FACEOCC_GDRIVE_URL" "$DOWNLOADS/FaceOcc.zip" "$FACEOCC_ARCHIVE_SIZE" "FaceOcc.zip"
    extract_faceocc
fi

if [[ ! -d "$CELEBAMASK_DIR/CelebA-HQ-img" || ! -d "$CELEBAMASK_DIR/CelebAMask-HQ-mask-anno" ]]; then
    download_archive "$CELEBAMASK_GDRIVE_URL" "$DOWNLOADS/CelebAMask-HQ.zip" "$CELEBAMASK_ARCHIVE_SIZE" "CelebAMask-HQ.zip"
    extract_celebamask
fi

preflight
rm -f "$DOWNLOADS/FaceOcc.zip" "$DOWNLOADS/CelebAMask-HQ.zip"
log "Setup complete"
