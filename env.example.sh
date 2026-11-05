# Local settings. Copy this file to env.sh (env.sh is not tracked by git) and edit it:
#
#     cp env.example.sh env.sh
#
# Every script in scripts/ and slurm/ sources env.sh if it exists.
# All paths have defaults inside the repository; set only what is different on your machine.

# --- conda --------------------------------------------------------------------------
# Name (or full path) of the conda environment that scripts/setup.sh creates and the
# other scripts activate.
export CONDA_ENV="${CONDA_ENV:-hoist}"

# --- CUDA ---------------------------------------------------------------------------
# The deformable-attention op of Mask2Former is compiled once by scripts/setup.sh.
# It needs nvcc 11.1 (the CUDA version of PyTorch 1.9.0+cu111) and gcc <= 10.
# export CUDA_HOME=/usr/local/cuda-11.1
# GPU architectures to compile for: 8.0 = A100, 8.6 = A40 / RTX 30xx, 7.5 = RTX 20xx.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}"

# --- data, weights, outputs (defaults shown) ----------------------------------------
# export DATA_DIR="$REPO/data"
# export RECORDINGS_DIR="$DATA_DIR/recordings"
# export ANNOTATIONS_DIR="$DATA_DIR/annotations"
# export JOB_FRAMES="$DATA_DIR/job_frames.json"
# export WEIGHTS_DIR="$REPO/weights"
# export OUTPUT_ROOT="$REPO/outputs"
# export HOIST_DIR="$REPO/third_party/HOISTFormer"

# --- example: Uni Bonn cluster "bender" (A100 nodes) --------------------------------
# source /software/easybuild-AMD_A100/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
# export CUDA_HOME=/software/easybuild-AMD_A100/software/CUDAcore/11.1.1
# export TORCH_CUDA_ARCH_LIST="8.0;8.6"
