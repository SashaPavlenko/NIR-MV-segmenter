"""All default locations in one place.

Every path can be overridden with an environment variable (see env.example.sh).
Defaults follow the repository layout:

    data/recordings/<run>/images/<device 1-4>/<left|right>/<timestamp_us>_<side>.bmp
    data/annotations/job_<id>.xml        CVAT for images 1.1 export, one file per view
    data/job_frames.json                 view (CVAT job) -> recording, camera, frame list
    weights/combo_nir/fold<k>.pth        final NIR model of fold k
    weights/combo_depth/fold<k>.pth      final NIR + disparity model of fold k
    third_party/HOISTFormer/             patched HOIST-Former checkout
    outputs/                             everything the scripts write
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _p(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


DATA_DIR = _p("DATA_DIR", REPO / "data")
RECORDINGS_DIR = _p("RECORDINGS_DIR", DATA_DIR / "recordings")
ANNOTATIONS_DIR = _p("ANNOTATIONS_DIR", DATA_DIR / "annotations")
JOB_FRAMES = _p("JOB_FRAMES", DATA_DIR / "job_frames.json")
WEIGHTS_DIR = _p("WEIGHTS_DIR", REPO / "weights")
OUTPUT_ROOT = _p("OUTPUT_ROOT", REPO / "outputs")
HOIST_DIR = _p("HOIST_DIR", REPO / "third_party" / "HOISTFormer")
GRID_DIR = _p("LEAP_GRID_DIR", REPO / "calibration" / "leap_grids")
FOLDS = _p("FOLDS_JSON", REPO / "splits" / "folds.json")
