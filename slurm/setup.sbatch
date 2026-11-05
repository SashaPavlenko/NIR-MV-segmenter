#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out
# Run from the repository root:  sbatch slurm/setup.sbatch [--pretrained]
# Add --partition=... for your cluster (bender: --partition=A100short).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/setup.sh "$@"
