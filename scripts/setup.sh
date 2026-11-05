#!/bin/bash
# Install everything that is not in this repository:
#   1. the conda environment (environment.yml),
#   2. HOIST-Former at the commit the thesis used, into third_party/HOISTFormer,
#   3. the thesis changes to HOIST-Former (hoistformer_patch/) and the model configs,
#   4. the deformable-attention CUDA op of Mask2Former,
#   5. (only with --pretrained) the original HOIST-Former weights, the start point of training.
#
#     bash scripts/setup.sh                 # inference only
#     bash scripts/setup.sh --pretrained    # also for training
#
# Run it on a machine with an NVIDIA GPU, or set FORCE_CUDA=1 and TORCH_CUDA_ARCH_LIST to compile
# the op without one. Steps that are already done are skipped, so it is safe to run again.
set -euo pipefail
source "$(dirname "$0")/common.sh"

HOIST_URL=https://github.com/SupreethN/HOISTFormer.git
HOIST_COMMIT=a229e83b7a164c0f6815c8fcb87b4bdbd05d6865
# HOIST-Former checkpoint from the HOIST-Former README (Google Drive)
HOIST_WEIGHTS_ID=1yLXCqt8mQroqQYDnncMvJ3HaOKMln8Xs
PRETRAINED=0
[ "${1:-}" = "--pretrained" ] && PRETRAINED=1

log "1/5 conda environment '$CONDA_ENV'"
command -v conda > /dev/null || { echo "conda not found (install Miniconda, or source conda.sh in env.sh)" >&2; exit 1; }
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1; print $NF}' | grep -qx "$CONDA_ENV"; then
    echo "    exists, not changed"
else
    if [[ "$CONDA_ENV" == */* ]]; then
        conda env create -f "$REPO/environment.yml" -p "$CONDA_ENV"
    else
        conda env create -f "$REPO/environment.yml" -n "$CONDA_ENV"
    fi
fi
conda activate "$CONDA_ENV"
python -c "import torch, detectron2; print('    torch', torch.__version__, '| detectron2', detectron2.__version__)"

log "2/5 HOIST-Former -> $HOIST_DIR"
if [ ! -d "$HOIST_DIR/.git" ]; then
    mkdir -p "$(dirname "$HOIST_DIR")"
    git clone "$HOIST_URL" "$HOIST_DIR"
fi
cd "$HOIST_DIR"
if [ -z "$(git status --porcelain --untracked-files=no)" ]; then
    git checkout -q "$HOIST_COMMIT"
fi
echo "    at $(git rev-parse --short HEAD)"

log "3/5 thesis patch and configs"
bash "$REPO/hoistformer_patch/apply_patch.sh" "$HOIST_DIR"

log "4/5 deformable-attention op"
if python -c "import MultiScaleDeformableAttention" 2> /dev/null && [ -z "${REBUILD_OP:-}" ]; then
    echo "    already installed (REBUILD_OP=1 to compile again)"
else
    [ -n "${CUDA_HOME:-}" ] || echo "    CUDA_HOME is not set: torch looks for nvcc on PATH" >&2
    (cd "$HOIST_DIR/mask2former/modeling/pixel_decoder/ops" && rm -rf build ./*.egg-info && sh make.sh > "$HOIST_DIR/build_op.log" 2>&1) \
        || { echo "    build failed, see $HOIST_DIR/build_op.log" >&2; exit 1; }
    python -c "import torch, MultiScaleDeformableAttention" && echo "    compiled"
fi

log "5/5 original HOIST-Former weights (training only)"
W="$HOIST_DIR/pretrained_models/trained_model.pth"
if [ -f "$W" ]; then
    echo "    present: $W"
elif [ "$PRETRAINED" = 1 ]; then
    mkdir -p "$(dirname "$W")"
    gdown "https://drive.google.com/uc?id=$HOIST_WEIGHTS_ID" -O "$W"
    ls -la "$W"
else
    echo "    skipped (run with --pretrained to download it; inference does not need it)"
fi

log "check"
cd "$HOIST_DIR"
python - <<'EOF'
import sys
sys.path.insert(0, ".")
import torch, detectron2, MultiScaleDeformableAttention  # noqa: F401
import mask2former, mask2former_video  # noqa: F401
from demo_video.predictor import VisualizationDemo  # noqa: F401
print("    imports ok, CUDA available:", torch.cuda.is_available())
EOF
log "setup done"
