# 3. Model weights

The trained weights are **not public** (they are trained on the private dataset). They are
available on request from the author, as one archive with ten files of 845 MB each:

```
weights/
├── combo_nir/          HOIST-Former, NIR input (three copies of the NIR frame)
│   ├── fold1.pth       trained without participant 1
│   ├── fold2.pth       trained without participant 2
│   ├── fold3.pth
│   ├── fold4.pth
│   └── fold5.pth
└── combo_depth/        HOIST-Former, NIR + disparity input
    ├── fold1.pth
    └── ...  fold5.pth
```

```bash
tar xf <weights archive> -C weights/
bash scripts/check_data.sh
```

Another location: set `WEIGHTS_DIR` in `env.sh`.

## Which file is used when

* **`scripts/reproduce_thesis.sh`** uses all ten: fold *k* predicts the recordings of
  participant *k*, which it has never seen. This gives the number of the thesis.
* **`scripts/predict.sh --fold k`** uses one NIR model and one depth model of the same fold.
  For one of the 15 thesis recordings, take the fold of its participant
  (`run_<p>_...` → `--fold <p>`), otherwise the scores are measured on training data.
  For new recordings any fold works; all five are trained on 4 of the 5 participants.

All models use the config `configs/hoist/v4_combo.yaml` (installed by `scripts/setup.sh`):
768 px input, 3× loss weight on scalpel, screwdriver and drill, brightness and contrast
augmentation. The two models differ only in their input.

## Files for training

Training starts from the original HOIST-Former checkpoint, not from these files.
`bash scripts/setup.sh --pretrained` downloads it (see [6_training.md](6_training.md)).
A newly trained fold is put into `weights/` with

```bash
bash scripts/collect_weights.sh outputs/train          # copies <model>/fold<k>/<last checkpoint>
```

## For the author: building the archive

The checkpoints of the thesis are in the training folders (one per model and fold); the file
to use is named in each `last_checkpoint`. `collect_weights.sh` copies them under the names above:

```bash
WEIGHTS_DIR=package/weights bash scripts/collect_weights.sh <folder with combo_nir/ and combo_depth/>
tar cf hoist_nir_weights.tar -C package/weights .
```
