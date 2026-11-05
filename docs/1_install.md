# 1. Installation

## Requirements

* Linux with an NVIDIA GPU. The final models take 768 × 768 px input. One inference process
  takes about 31 GB of GPU memory, training about 23 GB (measured on an A100 80 GB).
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).
* CUDA toolkit **11.1** with `nvcc` and a gcc up to version 10. Only needed once, to compile the
  deformable-attention op of Mask2Former. PyTorch itself brings its own CUDA runtime.
* About 10 GB of disk for the environment, the download caches and HOIST-Former (measured:
  environment 5.3 GB, conda and pip caches 3.4 GB, HOIST-Former with its weights 0.9 GB), plus
  the data (≈ 3 GB) and the weights (10 × 845 MB ≈ 8.5 GB for all fold models).

The exact versions (Python 3.8, PyTorch 1.9.0 + CUDA 11.1, Detectron2 0.6) are those of the
thesis experiments. HOIST-Former does not run with newer PyTorch versions without changes.

## Steps

```bash
git clone <this repository> && cd <repository>
cp env.example.sh env.sh
```

Open `env.sh` and set what is different on your machine. Usually only these two lines:

```bash
export CUDA_HOME=/usr/local/cuda-11.1          # folder with bin/nvcc of CUDA 11.1
export TORCH_CUDA_ARCH_LIST="8.0;8.6"          # GPU generations: 7.5 RTX 20xx, 8.0 A100, 8.6 A40/RTX 30xx
```

Then, on a machine with a GPU:

```bash
bash scripts/setup.sh                # for inference
bash scripts/setup.sh --pretrained   # also downloads the HOIST-Former weights, needed for training
```

`setup.sh` does five things. Each one is skipped if it is already done, so the script can be run again.

| step | what | where |
|---|---|---|
| 1 | conda environment from `environment.yml` (name from `CONDA_ENV`, default `hoist`) | conda envs |
| 2 | `git clone` of HOIST-Former, checkout of commit `a229e83` (the version of the thesis) | `third_party/HOISTFormer/` |
| 3 | thesis changes (`hoistformer_patch/hoistformer_thesis.patch`) and the model configs | same |
| 4 | compile the deformable-attention CUDA op (`make.sh` of Mask2Former) | conda env |
| 5 | with `--pretrained`: HOIST-Former checkpoint from Google Drive (886 MB) | `third_party/HOISTFormer/pretrained_models/trained_model.pth` |

At the end it imports everything and prints `imports ok, CUDA available: True`.

On a SLURM cluster, run it as a job: `sbatch slurm/setup.sbatch` (see [8_slurm.md](8_slurm.md)).

## What the patch changes in HOIST-Former

`hoistformer_patch/` holds the changes as a `git diff` against commit `a229e83` and, in
`files/`, the same changes as whole files.

| file | change |
|---|---|
| `mask2former_video/modeling/criterion.py` | `MODEL.MASK_FORMER.CLASS_LOSS_WEIGHTS`: weight per class on the class, mask and dice loss (3× for the tools) |
| `mask2former_video/data_video/augmentation.py` | `brightness_up` / `contrast_up`: one-sided brightness and contrast change in [1.0, 1.4] |
| `mask2former_video/data_video/dataset_mapper.py` | frame selection hook, class swap on horizontal flip |
| `mask2former_video/config.py`, `video_maskformer_model.py`, `datasets/hoist.py` | optional activity / camera embeddings (tested in the thesis, off in the final configs) |
| `mask2former_video/data_video/multiview_dataset_mapper.py` | sampler for multi-view clips (tested in the thesis, not used by the final method) |
| `mask2former/.../ms_deform_attn.py` | `im2col_step = 1` (less GPU memory) |
| `demo.py` | chunked demo |

Model configs (`configs/hoist/`): `my_finetune.yaml` → `v4_base.yaml` (384 px, 8 classes) →
`v4_combo.yaml` (final: 768 px, tool weights, augmentation).

## Problems

* **`nvcc fatal: Unsupported gpu architecture`** while compiling the op: `TORCH_CUDA_ARCH_LIST`
  contains a GPU generation that CUDA 11.1 does not know (for example 8.9 or 9.0). Remove it.
* **The op compiles, but `import MultiScaleDeformableAttention` fails on another GPU type.** Add
  that generation to `TORCH_CUDA_ARCH_LIST` and run `REBUILD_OP=1 bash scripts/setup.sh`.
* **No GPU on the machine where you install.** Set `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST`;
  the op is then compiled without a GPU.
* **The Google Drive download fails** (quota). Download `trained_model.pth` by hand from the
  [HOIST-Former README](https://github.com/SupreethN/HOISTFormer) and put it into
  `third_party/HOISTFormer/pretrained_models/`.
