# Multi-View Segmentation of Hands and Hand-Held Objects in Infrared Imaging

Code of the master thesis of Aleksandr Pavlenko (University of Bonn).

The method segments hands, haptic devices and hand-held tools in near-infrared (NIR) video
from four stereo sensors (eight views: four devices × left/right eye). It builds on
[HOIST-Former](https://github.com/SupreethN/HOISTFormer). Eight classes:
`hand_left`, `hand_right`, `haptic_left`, `haptic_right`, `scalpel`, `screwdriver`, `drill`, `object`.

**Final method** (thesis, Chapter 4):

1. **Disparity channel.** The two eyes of every device are rectified with the device grid and
   matched with SGM. The disparity map replaces the third copy of the NIR image.
2. **Two models.** HOIST-Former fine-tuned on NIR (`combo_nir`) and on NIR + disparity
   (`combo_depth`). 768 px input, 3× loss weight on the tools, brightness/contrast augmentation.
3. **Overlapping windows.** 16-frame windows every 8 frames, probabilities averaged.
4. **Ensemble** of the two models.
5. **Cross-view consensus.** The eight views of a recording vote on which classes are present
   in which frames (no calibration needed).

Result, leave-one-participant-out on 52 annotated views: **mIoU 0.656** over seven classes
(baseline, a single NIR model at 384 × 384 px without overlap and consensus: 0.588).

```
input frames ─┬─ NIR ─────────────────── combo_nir ───┐
              └─ NIR + disparity (SGM) ─ combo_depth ─┴─ ensemble ─ consensus (8 views) ─ masks / CVAT XML
```

## Quick start

```bash
git clone <this repository> && cd <repository>
cp env.example.sh env.sh            # edit: conda, CUDA_HOME (see docs/1_install.md)
bash scripts/setup.sh               # conda env + HOIST-Former + patch + CUDA op
# put the dataset into data/ and the weights into weights/ (both on request, see below)
bash scripts/check_data.sh          # are data/ and weights/ complete?
bash scripts/predict.sh --runs run_1_drill_3 --fold 1   # one recording -> outputs/predict/
bash scripts/reproduce_thesis.sh    # the thesis number (0.656) from the five fold models
```

## Documentation

| | |
|---|---|
| [docs/1_install.md](docs/1_install.md) | requirements, `scripts/setup.sh`, what it installs, problems |
| [docs/2_dataset.md](docs/2_dataset.md) | where the dataset goes, folder layout, `job_frames.json` |
| [docs/3_weights.md](docs/3_weights.md) | where the model weights go |
| [docs/4_inference.md](docs/4_inference.md) | running the final model on recordings, outputs |
| [docs/5_reproduce.md](docs/5_reproduce.md) | reproducing the thesis result, folds, metrics |
| [docs/6_training.md](docs/6_training.md) | training the models again |
| [docs/7_cvat.md](docs/7_cvat.md) | CVAT: installation, tasks, pre-annotation upload, export |
| [docs/8_slurm.md](docs/8_slurm.md) | running on a SLURM cluster |

## Data and weights

The recordings show the participants of the study, so the dataset and the trained weights are
**not public**. They are available on request from the author. The code expects them at

```
data/recordings/<run>/images/<device>/<side>/<timestamp>_<side>.bmp
data/annotations/job_<id>.xml
data/job_frames.json
weights/combo_nir/fold{1..5}.pth
weights/combo_depth/fold{1..5}.pth
```

Other locations can be set in `env.sh` (see [env.example.sh](env.example.sh)).

## Repository layout

```
scripts/            setup, prediction, reproduction, training (bash)
src/                Python code of the pipeline (one file per step)
hoistformer_patch/  changes to HOIST-Former (patch, the same changes as whole files, configs)
calibration/        LEAP_DISTORTION_MATRIX grids of the four devices (rectification)
splits/folds.json   leave-one-participant-out folds
cvat/               CVAT setup files and scripts (tasks, pre-annotation upload, export)
slurm/              sbatch wrappers of the scripts
data/  weights/     empty, see above
third_party/        HOIST-Former checkout (created by scripts/setup.sh)
outputs/            results (created by the scripts)
```

## Acknowledgements

[HOIST-Former](https://github.com/SupreethN/HOISTFormer) (Narasimhaswamy, Nguyen, Huang, Hoai,
CVPR 2024, MIT license) is downloaded into `third_party/HOISTFormer` by `scripts/setup.sh` and is
not part of this repository. `hoistformer_patch/` contains changes to its code. It builds on
Mask2Former and Detectron2.
