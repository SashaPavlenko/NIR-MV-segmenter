#!/bin/bash
#SBATCH --job-name=reproduce
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%x-%j.out
# Run from the repository root:  sbatch slurm/reproduce.sbatch
# Two inference processes share one GPU (about 31 GB each, fits an A100 80 GB); use PARALLEL=1
# for GPUs below 64 GB:  PARALLEL=1 sbatch slurm/reproduce.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PARALLEL="${PARALLEL:-2}" bash scripts/reproduce_thesis.sh
