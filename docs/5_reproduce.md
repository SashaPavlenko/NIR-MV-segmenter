# 5. Reproducing the thesis result

```bash
bash scripts/reproduce_thesis.sh                 # one inference process at a time
PARALLEL=2 bash scripts/reproduce_thesis.sh      # two at a time (an A100 80 GB fits two)
```

or on a cluster `sbatch slurm/reproduce.sbatch`.

## What it does

Leave-one-participant-out, as in the thesis (`splits/folds.json`):

1. disparity frames for the 120 views of the 15 recordings;
2. for every fold *k*: `weights/combo_nir/fold<k>.pth` and `weights/combo_depth/fold<k>.pth`
   predict the 24 views of participant *k* (who was not in their training data);
3. ensemble of the two models, overlapping windows;
4. cross-view consensus per recording;
5. scores on the 52 annotated views → `outputs/reproduce/eval/summary.md`.

## Expected result

Thesis, table "Leave-one-participant-out evaluation", row *full method (768²)*, mean IoU:

| hand l. | hand r. | hapt. l. | hapt. r. | scalpel | screwdr. | drill | tool_any | **mean (7 classes)** |
|---|---|---|---|---|---|---|---|---|
| 0.734 | 0.664 | 0.841 | 0.884 | 0.406 | 0.539 | 0.522 | 0.505 | **0.656** |

## Metrics

`src/evaluate.py`, the same definitions as in the thesis:

* per view and class (classes present in the ground truth of the view, plus `tool_any`, the
  union of the three tools): **IoU** per frame, averaged over the frames where ground truth or
  prediction is non-empty;
* **presence** per frame (class has a non-empty mask or not): precision, recall, F1 over all frames;
* every number is the mean over the views; the mean IoU of the method is the mean of the seven
  class values (without `object` and `tool_any`).

The GPU inference is not bit-exact between machines (another GPU node gives slightly different
window scores), so the last digit can differ. All steps after the inference are deterministic.

## How the code of this repository was checked

On the thesis cluster (A100 nodes), against the original experiment code:

* disparity frames, view index, ensemble masks, consensus masks and per-view scores computed by
  this repository from the original model outputs are **identical** to the original ones;
* a fresh installation with `scripts/setup.sh` and the original environment of the thesis give
  **bit-identical** model outputs on the same GPU node;
* the full `reproduce_thesis.sh` run from a fresh installation (A100 80 GB, `PARALLEL=2`,
  70 minutes) gave mean IoU **0.6559** on the 52 views. Every class value of the table above
  and every presence F1 value of the thesis table came out the same to three decimals.
