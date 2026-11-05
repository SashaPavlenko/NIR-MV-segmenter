# Shared setup for all scripts. Source it, do not run it.
# Loads env.sh, sets the default paths and activates the conda environment.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
# shellcheck disable=SC1091
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"

export CONDA_ENV="${CONDA_ENV:-hoist}"
export DATA_DIR="${DATA_DIR:-$REPO/data}"
export RECORDINGS_DIR="${RECORDINGS_DIR:-$DATA_DIR/recordings}"
export ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-$DATA_DIR/annotations}"
export JOB_FRAMES="${JOB_FRAMES:-$DATA_DIR/job_frames.json}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs}"
export HOIST_DIR="${HOIST_DIR:-$REPO/third_party/HOISTFormer}"
export FOLDS_JSON="${FOLDS_JSON:-$REPO/splits/folds.json}"
# the final models: HOIST-Former, 768 px input, 3x loss weight on the tools, brightness/contrast
export CONFIG="${CONFIG:-configs/hoist/v4_combo.yaml}"
SRC="$REPO/src"

activate_env() {
    if [ "${CONDA_PREFIX:-}" != "" ] && { [ "$(basename "$CONDA_PREFIX")" = "$CONDA_ENV" ] || [ "$CONDA_PREFIX" = "$CONDA_ENV" ]; }; then
        return 0
    fi
    if ! command -v conda > /dev/null; then
        echo "conda not found: install Miniconda or source conda.sh in env.sh" >&2
        return 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
}

# weights_of <model: combo_nir | combo_depth> <fold 1-5>
weights_of() {
    local w="$WEIGHTS_DIR/$1/fold$2.pth"
    if [ ! -f "$w" ]; then
        echo "missing weights: $w (see docs/3_weights.md)" >&2
        return 1
    fi
    echo "$w"
}

# fold_field <fold 1-5> <test_jobs | infer_jobs>  -> space-separated job ids from splits/folds.json
fold_field() {
    python -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['fold'+sys.argv[2]][sys.argv[3]])))" \
        "$FOLDS_JSON" "$1" "$2"
}

log() { echo "[$(date +%H:%M:%S)] $*"; }
