"""Build a view list (same format as job_frames.json) for recordings that are not in CVAT.

    python src/make_views.py --recordings data/recordings --runs run_7_drill_1 --step 3 \
        --out outputs/new/views.json

A view is one eye of one device of one recording: <run>/images/<device>/<left|right>/.
Every --step-th frame (sorted by timestamp) is kept, as CVAT does with frame_step when a task
is created (cvat/create_tasks.py). Views get ids --first-id, --first-id + 1, ... and the task
name "<run>__cam<device>_<side>" that the rest of the pipeline expects.

When the frames are already in CVAT, use cvat/export_job_frames.py instead: then the ids are
the real CVAT job ids and the predictions can be uploaded into those jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=paths.RECORDINGS_DIR)
    ap.add_argument("--runs", nargs="*", default=None, help="default: every folder in --recordings")
    ap.add_argument("--step", type=int, default=1, help="keep every step-th frame")
    ap.add_argument("--first-id", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    runs = a.runs or sorted(p.name for p in a.recordings.iterdir() if (p / "images").is_dir())
    views, jid = [], a.first_id
    for run in runs:
        for dev in (1, 2, 3, 4):
            for side in ("left", "right"):
                d = a.recordings / run / "images" / str(dev) / side
                files = sorted(d.glob(f"*_{side}.bmp"), key=lambda p: int(p.name.split("_")[0])) if d.is_dir() else []
                if not files:
                    print(f"skip {run} cam{dev}_{side}: no frames in {d}")
                    continue
                kept = list(range(0, len(files), a.step))
                views.append({"job_id": jid, "task_name": f"{run}__cam{dev}_{side}",
                              "frame_count": len(kept), "frame_ids": kept,
                              "frame_names": [files[i].name for i in kept],
                              "frame_filter": f"step={a.step}" if a.step > 1 else ""})
                jid += 1
    if not views:
        sys.exit("no views found")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(views, indent=1))
    print(f"{len(views)} views of {len(runs)} recordings -> {a.out}")


if __name__ == "__main__":
    main()
