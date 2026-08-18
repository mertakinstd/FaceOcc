#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="faceocc"
THREEDDFA_REPO="https://github.com/cleardusk/3DDFA_V2.git"
THREEDDFA_REF="v0.12"
THREEDDFA_COMMIT="70e7cde"
TDDFA_ID="1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk"
TDDFA_SIZE="13184521"
TDDFA_SHA256="1c0a8acd50db28987773324a9b2b816361468e3aa13cb6b212c911b889e08c3e"
FACEBOXES_ID="1pccQOvYqKh3iCEHc5tSWx2-1fhgxs6rh"
FACEBOXES_SIZE="4063834"
FACEBOXES_SHA256="260f4916731c781ab71080ef00f21f32a6566bdd86ccb3c8b0fbce269e829ce1"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TOOLS="$ROOT/.tools"
MAMBA_ROOT_PREFIX="$ROOT/.micromamba"
MICROMAMBA="$TOOLS/micromamba/bin/micromamba"
THREEDDFA="$TOOLS/3DDFA_V2"
RAW="$ROOT/data/raw/CelebAMask-HQ"
LANDMARKS="$ROOT/data/prepared/landmarks"
ALIGNED_IMAGES="$ROOT/Dataset/CelebA-HQ-align"
ALIGNED_MASKS="$ROOT/Dataset/CelebAMask-HQ-align"

export MAMBA_ROOT_PREFIX MAMBA_NO_BANNER=1

log() { printf '\n[FaceOcc prepare] %s\n' "$*"; }
die() { printf '\n[FaceOcc prepare] ERROR: %s\n' "$*" >&2; exit 1; }
run() { "$MICROMAMBA" run -n "$ENV_NAME" "$@"; }

[[ "$#" -eq 0 ]] || die "prepare.sh takes no arguments."
for command in git sha256sum stat find; do
    command -v "$command" >/dev/null || die "Required command not found: $command"
done
[[ -x "$MICROMAMBA" ]] || die "Run ./setup.sh first."
[[ -d "$RAW/CelebA-HQ-img" && -d "$RAW/CelebAMask-HQ-mask-anno" ]] || die "Raw CelebAMask-HQ data is missing."
[[ ! -e "$LANDMARKS" && ! -e "$ALIGNED_IMAGES" && ! -e "$ALIGNED_MASKS" ]] || die "Prepared outputs already exist; remove them before running preparation."

if [[ ! -d "$THREEDDFA/.git" ]]; then
    [[ ! -e "$THREEDDFA" ]] || die "$THREEDDFA exists but is not a Git checkout."
    log "Fetching 3DDFA_V2"
    git clone --quiet --depth 1 --branch "$THREEDDFA_REF" "$THREEDDFA_REPO" "$THREEDDFA"
fi
[[ "$(git -C "$THREEDDFA" rev-parse HEAD)" == "$THREEDDFA_COMMIT"* ]] || die "Unexpected 3DDFA_V2 revision."
git -C "$THREEDDFA" diff --quiet || die "3DDFA_V2 checkout is modified."

fetch_model() {
    local id="$1" output="$2" size="$3" sha="$4"
    mkdir -p "$(dirname "$output")"
    if [[ ! -f "$output" ]]; then
        run python -m gdown "https://drive.google.com/uc?id=$id" -O "$output"
    fi
    [[ "$(stat -c '%s' "$output")" == "$size" ]] || die "Invalid model artifact: $output"
    [[ "$(sha256sum "$output" | cut -d' ' -f1)" == "$sha" ]] || die "Invalid model artifact: $output"
}

fetch_model "$TDDFA_ID" "$THREEDDFA/weights/mb1_120x120.onnx" "$TDDFA_SIZE" "$TDDFA_SHA256"
fetch_model "$FACEBOXES_ID" "$THREEDDFA/FaceBoxes/weights/FaceBoxesProd.onnx" "$FACEBOXES_SIZE" "$FACEBOXES_SHA256"

log "Generating 3DDFA landmarks"
run python -m face_align.generate_3ddfa_landmarks

log "Aligning CelebAMask-HQ"
run python -m face_align.process_CelebAMaskHQ

landmark_count="$(find "$LANDMARKS" -maxdepth 1 -type f -name '*.npy' | wc -l | tr -d ' ')"
image_count="$(find "$ALIGNED_IMAGES" -maxdepth 1 -type f | wc -l | tr -d ' ')"
mask_count="$(find "$ALIGNED_MASKS" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
[[ "$landmark_count" -gt 0 ]] || die "No landmarks were generated."
[[ "$image_count" -eq "$landmark_count" ]] || die "Aligned image count does not match landmark count."
[[ "$mask_count" -eq "$landmark_count" ]] || die "Aligned mask count does not match landmark count."

run python - <<'PY'
from pathlib import Path
from Dataset.dataset import FaceMask, COFW_test, mask_root, occ_root

required = Path(occ_root) / 'CelebAHQ'
missing = [path.stem for path in required.iterdir() if not (Path(mask_root) / f'{path.stem}.png').is_file()]
if missing:
    raise RuntimeError(f'{len(missing)} FaceOcc CelebAHQ samples lack aligned masks')

train = FaceMask()
valid = COFW_test()
image, mask = train[0]
assert image.shape == (3, 256, 256)
assert mask.shape in {(1, 256, 256), (256, 256)}
assert len(valid) > 0
PY

log "Dataset preparation complete: $landmark_count/30000 CelebA-HQ samples aligned"
