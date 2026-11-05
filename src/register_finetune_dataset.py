"""
register_finetune_dataset.py — Registers our 8-class CVAT-derived datasets
into detectron2's DatasetCatalog under names distinct from the repo's
built-in 1-class "hoist_train/test/valid" (mask2former_video/data_video/
datasets/hoist.py), so both coexist without touching that file.

Import this module (for its side effects) before building the trainer, e.g.:
    import register_finetune_dataset  # noqa: F401

Dataset names registered (paths are read from env vars so the same module
works for the smoke test / QC-train / final full-train JSONs):
    hoist_finetune_train    <- FINETUNE_TRAIN_JSON  (default: train_minus_holdout.json)
    hoist_finetune_holdout  <- FINETUNE_HOLDOUT_JSON (default: holdout.json)
"""

import os
import sys
from pathlib import Path

from detectron2.data import MetadataCatalog

from mask2former_video.data_video.datasets.hoist import register_hoist_instances

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories import CATEGORIES  # noqa: E402

DATA_DIR = Path(os.environ.get(
    "FINETUNE_DATA_DIR", str(Path(__file__).resolve().parents[1] / "outputs" / "train_data")
))


def _meta():
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(CATEGORIES))}
    return {
        "thing_classes": CATEGORIES,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    }


_SPLITS = {
    "hoist_finetune_train": os.environ.get(
        "FINETUNE_TRAIN_JSON", str(DATA_DIR / "train_minus_holdout.json")
    ),
    "hoist_finetune_holdout": os.environ.get(
        "FINETUNE_HOLDOUT_JSON", str(DATA_DIR / "holdout.json")
    ),
    "hoist_finetune_smoke": os.environ.get(
        "FINETUNE_SMOKE_JSON", str(DATA_DIR / "smoke.json")
    ),
    "hoist_finetune_all": os.environ.get(
        "FINETUNE_ALL_JSON", str(DATA_DIR / "all.json")
    ),
}

for _name, _json_file in _SPLITS.items():
    if not Path(_json_file).exists():
        continue
    # image_root="" because file_names in our JSON are already absolute paths.
    register_hoist_instances(_name, _meta(), _json_file, "")
    MetadataCatalog.get(_name).evaluator_type = "ytvis"
