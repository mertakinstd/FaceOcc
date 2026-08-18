#!/usr/bin/env bash
set -Eeuo pipefail

# FaceOcc bootstrap for Linux.
# - installs a repo-local micromamba binary and root prefix
# - creates the environment from environment.yml
# - loads local setup configuration from .env (without executing it as shell code)
# - downloads FaceOcc.zip and CelebAMask-HQ.zip from configured Google Drive URLs
# - extracts datasets into deterministic locations
#
# Usage:
#   cp .env.example .env
#   # Edit .env, review the CelebAMask-HQ license, then set
#   # CELEBAMASK_LICENSE_ACCEPTED=1 if you accept it.
#   ./setup.sh

FACEOCC_ARCHIVE_SIZE="538014369"
CELEBAMASK_ARCHIVE_SIZE="3153930546"
GDOWN_VERSION="5.2.0"
ENV_NAME="faceocc"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$SCRIPT_DIR"
TOOLS_DIR="$REPO_ROOT/.tools"
MAMBA_ROOT_PREFIX="$REPO_ROOT/.micromamba"
MICROMAMBA="$TOOLS_DIR/micromamba/bin/micromamba"
DOWNLOAD_DIR="$REPO_ROOT/.cache/faceocc-downloads"
EXTRACT_DIR="$REPO_ROOT/.cache/faceocc-extract"
FACEOCC_DEST="$REPO_ROOT/Dataset/FaceOcc"
CELEBAMASK_DEST="$REPO_ROOT/data/raw/CelebAMask-HQ"

export MAMBA_ROOT_PREFIX
export MAMBA_NO_BANNER=1

log()  { printf '\n[FaceOcc setup] %s\n' "$*"; }
die()  { printf '\n[FaceOcc setup] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

usage() {
    cat <<'USAGE'
Usage: ./setup.sh

Configuration is read from .env in the repository root. Start with:
  cp .env.example .env

Required .env values:
  FACEOCC_GDRIVE_URL=...
  CELEBAMASK_GDRIVE_URL=...
  CELEBAMASK_LICENSE_ACCEPTED=1

Optional:
  KEEP_ARCHIVES=1

CelebAMask-HQ is restricted to non-commercial research/educational use by
its upstream dataset agreement. Set CELEBAMASK_LICENSE_ACCEPTED=1 only after
reviewing and accepting that agreement.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ "$#" -eq 0 ]] || die "setup.sh takes no arguments; configure it through .env (use --help for details)."

trim_env_value() {
    local value="$1"
    value="${value%$'\r'}"
    # Strip matching single or double quotes without eval/source.
    if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] || \
           [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
            value="${value:1:${#value}-2}"
        fi
    fi
    printf '%s' "$value"
}

load_env_file() {
    local env_file="$REPO_ROOT/.env"
    [[ -f "$env_file" ]] || die \
        ".env not found. Run 'cp .env.example .env', edit it, then rerun setup.sh."

    local line key value line_no=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_no += 1))
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" == *=* ]] || die "Invalid .env entry at line $line_no: expected KEY=VALUE."

        key="${line%%=*}"
        value="${line#*=}"
        key="${key//[[:space:]]/}"
        value="$(trim_env_value "$value")"

        case "$key" in
            FACEOCC_GDRIVE_URL) FACEOCC_GDRIVE_URL="$value" ;;
            CELEBAMASK_GDRIVE_URL) CELEBAMASK_GDRIVE_URL="$value" ;;
            CELEBAMASK_LICENSE_ACCEPTED) CELEBAMASK_LICENSE_ACCEPTED="$value" ;;
            KEEP_ARCHIVES) KEEP_ARCHIVES="$value" ;;
            *) die "Unknown .env key '$key' at line $line_no." ;;
        esac
    done < "$env_file"
}

load_env_file

[[ -n "${FACEOCC_GDRIVE_URL:-}" ]] || die "FACEOCC_GDRIVE_URL is empty in .env."
[[ -n "${CELEBAMASK_GDRIVE_URL:-}" ]] || die "CELEBAMASK_GDRIVE_URL is empty in .env."
[[ "${CELEBAMASK_LICENSE_ACCEPTED:-0}" == "1" ]] || die \
    "CelebAMask-HQ license not accepted. Review it, then set CELEBAMASK_LICENSE_ACCEPTED=1 in .env."

[[ -f "$REPO_ROOT/environment.yml" ]] || die "environment.yml not found in repo root: $REPO_ROOT"
[[ -f "$REPO_ROOT/requirements.txt" ]] || die "requirements.txt not found in repo root: $REPO_ROOT"
[[ -f "$REPO_ROOT/train.py" ]] || die "train.py not found; run setup.sh from the FaceOcc repository root."

need curl
need tar
need unzip
need find
need awk
need wc
need stat

case "$(uname -s)" in
    Linux) ;;
    *) die "This setup script currently supports Linux only." ;;
esac

case "$(uname -m)" in
    x86_64|amd64) MAMBA_PLATFORM="linux-64" ;;
    aarch64|arm64) MAMBA_PLATFORM="linux-aarch64" ;;
    *) die "Unsupported CPU architecture: $(uname -m)" ;;
esac

install_micromamba() {
    if [[ -x "$MICROMAMBA" ]]; then
        log "Using existing repo-local micromamba: $MICROMAMBA"
        return
    fi

    log "Installing repo-local micromamba ($MAMBA_PLATFORM)"
    mkdir -p "$TOOLS_DIR/micromamba"
    (
        cd "$TOOLS_DIR/micromamba"
        curl --fail --location --silent --show-error \
            "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" \
            | tar -xj bin/micromamba
    )
    [[ -x "$MICROMAMBA" ]] || die "micromamba installation failed."
}

create_environment() {
    if "$MICROMAMBA" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "Micromamba environment '$ENV_NAME' already exists; updating from environment.yml"
        "$MICROMAMBA" env update -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml" -y
    else
        log "Creating Micromamba environment '$ENV_NAME'"
        (
            cd "$REPO_ROOT"
            "$MICROMAMBA" create -f environment.yml -y
        )
    fi

    # gdown is a setup/download utility rather than a model runtime dependency.
    if ! "$MICROMAMBA" run -n "$ENV_NAME" python -c 'import gdown' >/dev/null 2>&1; then
        log "Installing setup-only downloader gdown==$GDOWN_VERSION"
        "$MICROMAMBA" run -n "$ENV_NAME" python -m pip install \
            --disable-pip-version-check --no-cache-dir "gdown==$GDOWN_VERSION"
    fi
}

download_gdrive() {
    local url="$1"
    local output="$2"
    local expected_size="$3"
    local label="$4"

    mkdir -p "$(dirname "$output")"

    if [[ -f "$output" ]]; then
        local current_size
        current_size="$(stat -c '%s' "$output")"
        if [[ "$current_size" == "$expected_size" ]]; then
            log "$label archive already downloaded and size matches; reusing it"
            return
        fi
        if (( current_size < expected_size )); then
            log "$label archive is partial ($current_size/$expected_size bytes); attempting resume"
        else
            log "$label archive is larger than expected ($current_size > $expected_size); removing it"
            rm -f "$output"
        fi
    fi

    log "Downloading $label"
    "$MICROMAMBA" run -n "$ENV_NAME" python -m gdown \
        --continue --fuzzy "$url" \
        -O "$output"

    local actual_size
    actual_size="$(stat -c '%s' "$output")"
    [[ "$actual_size" == "$expected_size" ]] || die \
        "$label download size mismatch: got $actual_size bytes, expected $expected_size bytes."

    unzip -tq "$output" >/dev/null || die "$label archive failed ZIP integrity test."
}

find_dir_named() {
    local root="$1"
    local name="$2"
    find "$root" -type d -name "$name" -print -quit
}

extract_faceocc() {
    if [[ -d "$FACEOCC_DEST" ]]; then
        if [[ -d "$FACEOCC_DEST/CelebAHQ" && -d "$FACEOCC_DEST/COFW_test" ]]; then
            log "FaceOcc dataset already present: $FACEOCC_DEST"
            return
        fi
        die "Existing $FACEOCC_DEST does not look like a complete FaceOcc dataset. Move/remove it before retrying."
    fi

    local archive="$DOWNLOAD_DIR/FaceOcc.zip"
    local tmp="$EXTRACT_DIR/faceocc"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    log "Extracting FaceOcc"
    unzip -q "$archive" -d "$tmp"

    local src
    src="$(find_dir_named "$tmp" "FaceOcc")"
    if [[ -z "$src" ]]; then
        # Some archives may contain the FaceOcc children directly at archive root.
        if [[ -d "$tmp/CelebAHQ" && -d "$tmp/COFW_test" ]]; then
            src="$tmp"
        else
            die "Could not locate FaceOcc dataset root inside FaceOcc.zip."
        fi
    fi

    [[ -d "$src/CelebAHQ" ]] || die "FaceOcc/CelebAHQ missing after extraction."
    [[ -d "$src/COFW_test" ]] || die "FaceOcc/COFW_test missing after extraction."

    mkdir -p "$(dirname "$FACEOCC_DEST")"
    mv "$src" "$FACEOCC_DEST"
    rm -rf "$tmp"
}

extract_celebamask() {
    if [[ -d "$CELEBAMASK_DEST" ]]; then
        if [[ -d "$CELEBAMASK_DEST/CelebA-HQ-img" && -d "$CELEBAMASK_DEST/CelebAMask-HQ-mask-anno" ]]; then
            log "CelebAMask-HQ already present: $CELEBAMASK_DEST"
            return
        fi
        die "Existing $CELEBAMASK_DEST does not look complete. Move/remove it before retrying."
    fi

    local archive="$DOWNLOAD_DIR/CelebAMask-HQ.zip"
    local tmp="$EXTRACT_DIR/celebamask"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    log "Extracting CelebAMask-HQ"
    unzip -q "$archive" -d "$tmp"

    local img_dir src
    img_dir="$(find_dir_named "$tmp" "CelebA-HQ-img")"
    [[ -n "$img_dir" ]] || die "CelebA-HQ-img was not found inside CelebAMask-HQ.zip."
    src="$(dirname "$img_dir")"

    [[ -d "$src/CelebAMask-HQ-mask-anno" ]] || die \
        "CelebAMask-HQ-mask-anno missing next to CelebA-HQ-img."

    mkdir -p "$(dirname "$CELEBAMASK_DEST")"
    mv "$src" "$CELEBAMASK_DEST"
    rm -rf "$tmp"
}

preflight() {
    log "Running environment and dataset preflight"

    "$MICROMAMBA" run -n "$ENV_NAME" python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; FaceOcc training is GPU-only.")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

    local celeb_images faceocc_files
    celeb_images="$(find "$CELEBAMASK_DEST/CelebA-HQ-img" -maxdepth 1 -type f | wc -l | tr -d ' ')"
    faceocc_files="$(find "$FACEOCC_DEST" -type f | wc -l | tr -d ' ')"

    [[ "$celeb_images" -eq 30000 ]] || die \
        "Expected 30,000 CelebA-HQ images, found $celeb_images in $CELEBAMASK_DEST/CelebA-HQ-img."

    printf '\nDataset preflight:\n'
    printf '  FaceOcc files:       %s\n' "$faceocc_files"
    printf '  CelebA-HQ images:    %s\n' "$celeb_images"
    printf '  FaceOcc path:        %s\n' "$FACEOCC_DEST"
    printf '  CelebAMask-HQ path:  %s\n' "$CELEBAMASK_DEST"
}

main() {
    mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

    install_micromamba
    create_environment

    if [[ -d "$FACEOCC_DEST/CelebAHQ" && -d "$FACEOCC_DEST/COFW_test" ]]; then
        log "FaceOcc dataset already present; skipping its download"
    else
        download_gdrive \
            "$FACEOCC_GDRIVE_URL" \
            "$DOWNLOAD_DIR/FaceOcc.zip" \
            "$FACEOCC_ARCHIVE_SIZE" \
            "FaceOcc.zip"
        extract_faceocc
    fi

    if [[ -d "$CELEBAMASK_DEST/CelebA-HQ-img" && -d "$CELEBAMASK_DEST/CelebAMask-HQ-mask-anno" ]]; then
        log "CelebAMask-HQ already present; skipping its download"
    else
        download_gdrive \
            "$CELEBAMASK_GDRIVE_URL" \
            "$DOWNLOAD_DIR/CelebAMask-HQ.zip" \
            "$CELEBAMASK_ARCHIVE_SIZE" \
            "CelebAMask-HQ.zip"
        extract_celebamask
    fi
    preflight

    if [[ "${KEEP_ARCHIVES:-0}" != "1" ]]; then
        log "Removing downloaded ZIP archives (set KEEP_ARCHIVES=1 to retain them)"
        rm -f "$DOWNLOAD_DIR/FaceOcc.zip" "$DOWNLOAD_DIR/CelebAMask-HQ.zip"
    fi

    cat <<EOF_DONE

[FaceOcc setup] Base setup complete.

Repo-local micromamba:
  $MICROMAMBA

Environment root:
  $MAMBA_ROOT_PREFIX

To open an activated shell:
  export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
  eval "\$($MICROMAMBA shell hook -s bash)"
  micromamba activate $ENV_NAME

Raw datasets are ready. Training is NOT ready yet: CelebAMask-HQ still needs
3DDFA_V2 landmark detection + FaceOcc alignment preprocessing to generate:
  Dataset/CelebA-HQ-align/
  Dataset/CelebAMask-HQ-align/
EOF_DONE
}

main "$@"
