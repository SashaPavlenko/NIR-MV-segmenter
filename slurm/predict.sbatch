#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out
# Run from the repository root, with the options of scripts/predict.sh:
#     sbatch slurm/predict.sbatch --runs "run_1_drill_3" --fold 1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/predict.sh --workers 8 "$@"
