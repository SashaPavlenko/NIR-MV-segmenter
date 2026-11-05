#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%x-%j.out
# Run from the repository root:
#     MODEL=combo_nir sbatch slurm/train.sbatch           # five folds, one after another
#     MODEL=combo_nir PARALLEL=3 sbatch slurm/train.sbatch  # three folds at once (about 23 GB each)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/train_lopo.sh
