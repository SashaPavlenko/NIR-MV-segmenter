# 6. Training

```bash
bash scripts/setup.sh --pretrained                       # once: original HOIST-Former weights
MODEL=combo_nir   bash scripts/train_lopo.sh             # five folds, NIR input
MODEL=combo_depth bash scripts/train_lopo.sh             # five folds, NIR + disparity input
bash scripts/collect_weights.sh outputs/train            # -> weights/<model>/fold<k>.pth
```

`FOLDS="1 3"` trains only some folds. `FOLDS=all` trains one model on all 52 annotated views
(→ `outputs/train/<MODEL>/foldall/`, after `collect_weights.sh` → `weights/<MODEL>/foldall.pth`,
used with `predict.sh --fold all`). Such a model is meant for new data; it has no held-out score.

## What the script does per fold

1. **Training views** = all annotated views (`data/annotations/job_*.xml`) minus the
   `test_jobs` of the fold.
2. `src/cvat_to_ytvis.py` converts them to the YTVIS JSON that HOIST-Former reads (one video per
   view, one track per class, the masks of a label merged per frame). For `combo_depth` the
   frame paths are switched to the disparity frames (made first by `src/precompute_depth.py`).
3. `src/train_finetune.py` runs HOIST-Former's `train_net_video.py` with the config
   `configs/hoist/v4_combo.yaml`, starting from `pretrained_models/trained_model.pth`.
4. Early stopping on the smoothed training loss: stop when it has not improved by 0.01 for 800
   iterations (checked every 50, not before iteration 300). At most 5000 iterations.

Training settings (from the configs in `hoistformer_patch/configs/hoist/`):

| | |
|---|---|
| model | HOIST-Former: Swin-L backbone, 200 queries, 8 classes |
| input | 768 × 768 px (frames are 384 × 384, upscaled) |
| clip | 2 frames of one view within a range of 20 frames (HOIST-Former default) |
| batch | 1 clip, mixed precision (AMP) |
| optimizer | AdamW, weight decay 0.05, backbone learning rate × 0.1, gradient clipping |
| learning rate | 1e-5, × 0.1 at iteration 3000, 10 warm-up iterations |
| iterations | at most 5000, checkpoint every 250 |
| loss weights per class | 3 for scalpel, screwdriver, drill, 1 for the others |
| augmentation | horizontal flip per clip, brightness and contrast with a factor in [1.0, 1.4] |

Output: `outputs/train/<MODEL>/fold<k>/` with `train.log`, `metrics.json` and the checkpoints.
The checkpoint to use is the one in `last_checkpoint`. The two newest periodic checkpoints are kept.

Training is not deterministic. A new run gives similar, not identical, weights.

## Short test

```bash
MODEL=combo_nir FOLDS=1 EXTRA_OPTS="SOLVER.MAX_ITER 20 SOLVER.CHECKPOINT_PERIOD 10" bash scripts/train_lopo.sh
```

runs 20 iterations of fold 1 and writes a checkpoint, to check that training works.
