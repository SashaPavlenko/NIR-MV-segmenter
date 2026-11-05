# 4. Running the final model

```bash
bash scripts/predict.sh --runs "run_1_drill_3" --fold 1 --out outputs/predict
```

This runs the whole final method on every view of the given recordings. Options:

| option | default | meaning |
|---|---|---|
| `--runs "A B"` | all recordings in the view list | recordings to process (always all eight views of each) |
| `--fold K` | 1 | which pair of models (`weights/combo_*/fold<K>.pth`), see [3_weights.md](3_weights.md) |
| `--views FILE` | `data/job_frames.json` | view list, from CVAT or from `src/make_views.py` |
| `--recordings DIR` | `data/recordings` | frames |
| `--out DIR` | `outputs/predict` | output folder |
| `--workers N` | 8 | CPU processes for the CPU steps |

Steps that already have their output are skipped. To start again, delete the output folder.

## What happens

| step | script | output in `--out` |
|---|---|---|
| 1. disparity frames: rectify both eyes with the device grid, SGM, map back | `src/precompute_depth.py` | `depth_frames/` (PNG: disparity, NIR, NIR) |
| 2. NIR model, 16-frame windows every 8 frames | `src/infer_soft.py` | `soft_nir/job_<id>.npz` |
| 2. NIR + disparity model, same windows | `src/infer_soft.py` | `soft_depth/job_<id>.npz` |
| 3. average the windows, average the two models, pixel > 0.5, drop small blobs of tools and haptics | `src/ensemble.py` | `ensemble/job_<id>.npz` |
| 4. cross-view consensus of the eight views | `src/build_view_index.py`, `src/crossview_consensus.py` | `view_index.json`, `consensus/job_<id>.npz`, `consensus_report.json` |
| 5. CVAT XML | `src/postprocess_and_export.py` | `cvat_xml/job_<id>.xml` |
| 6. scores, if views are annotated | `src/evaluate.py` | `eval/summary.md`, `eval/scores.json` |

One recording (8 views, 132-134 frames each) takes about 7 minutes on an A100: most of it is
the two models (about 35 s per view and model), and one inference process takes about 31 GB of
GPU memory.

The final masks are `consensus/job_<id>.npz`: `masks` (instances × frames × 384 × 384, uint8),
`labels` (class index, order of `src/categories.py`), `scores`. `cvat_xml/` holds the same
masks as CVAT for images 1.1 track XML, ready for `cvat/upload_preannotations.py`
(see [7_cvat.md](7_cvat.md)).

### The consensus rules

The four devices see the same scene, so the eight views must agree on which objects exist.
`crossview_consensus.py` (defaults = final method):

* **class vote**: a view "claims" a class if the class is present in at least 20 % of its
  frames. A class claimed by fewer than 30 % of the views is dropped in every view;
* **frame vote**: per moment (frames of all views within 20 ms). Only the views that show the
  class somewhere in the recording take part. If just one of them has the class at this moment
  and that is at most 25 % of them, the detection is dropped. If at least 60 % of them have it,
  the frame is marked as a gap in the others;
* **temporal fill**: a gap is filled from the nearest frame of the same view (at most 3 frames
  away). A class that a view never shows is never added to it.

Recordings with fewer than 3 predicted views are passed through unchanged.

`consensus_report.json` lists what each rule changed per recording and view.

## New recordings

Recordings that are not in `job_frames.json`:

```bash
python src/make_views.py --recordings /path/to/recordings --runs run_7_drill_1 --step 3 --out outputs/new/views.json
bash scripts/predict.sh --views outputs/new/views.json --recordings /path/to/recordings --fold 1 --out outputs/new
```

`--step 3` takes every 3rd frame (30 fps from 90 fps), as for the thesis data. The models
were trained on such frames.

If the recordings should be corrected in CVAT afterwards, create the CVAT tasks first and
use the real job ids (see [7_cvat.md](7_cvat.md), "Pre-annotation").
