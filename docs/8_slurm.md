# 8. Running on a SLURM cluster

`slurm/` holds thin `sbatch` wrappers around the scripts. Submit them **from the repository
root** (they `cd` to the submit folder and read `env.sh` there). Add the partition of your
cluster with `--partition=...` (for example `--partition=A100short` on the Uni Bonn cluster bender).

```bash
sbatch slurm/setup.sbatch --pretrained                         # install (needs a GPU node for the CUDA op)
sbatch slurm/predict.sbatch --runs "run_1_drill_3" --fold 1    # options of scripts/predict.sh
sbatch slurm/reproduce.sbatch                                  # thesis number, PARALLEL=2 by default
MODEL=combo_nir sbatch slurm/train.sbatch                      # five folds one after another
```

Log: `slurm-<name>-<jobid>.out` in the repository root.

| job | GPU memory | time on one A100 (measured) |
|---|---|---|
| `setup.sbatch` | – | about 4 min (with a fast network) |
| `predict.sbatch`, one recording (8 views) | about 31 GB | about 7 min |
| `reproduce.sbatch`, 120 views, `PARALLEL=2` | 2 × 31 GB | about 70 min |
| `train.sbatch`, one fold | about 23 GB | about 0.6 s per iteration, at most 5000 iterations (early stopping) |

## Notes for bender (Uni Bonn)

* `env.sh`:

  ```bash
  source /software/easybuild-AMD_A100/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
  export CUDA_HOME=/software/easybuild-AMD_A100/software/CUDAcore/11.1.1
  export TORCH_CUDA_ARCH_LIST="8.0;8.6"
  ```

* The compute nodes have no `git`; `environment.yml` installs it into the conda environment,
  so `setup.sbatch` works on a compute node.
* The home quota is 300 GB. The conda caches (`~/.conda/pkgs`) and the training checkpoints
  (845 MB each) fill it fast. When the quota is exceeded, no file can be written at all and jobs
  fail with `Disk quota exceeded`. `conda clean --all` frees the conda caches.
* Training runs that stop early can keep running in their data loader workers;
  `train_lopo.sh` ends them once the log reports the total training time.
