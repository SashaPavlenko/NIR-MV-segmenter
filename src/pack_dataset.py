"""Build the dataset folder that the repository expects, from the original files.

    python src/pack_dataset.py --recordings /path/to/dataset_12-05 --annotations /path/to/cvat_xml \
        --job-frames /path/to/job_frames.json --out /path/to/package/data [--runs-from splits/folds.json]

Copies (does not move) into --out:
    recordings/<run>/images/<device>/<side>/<frame>.bmp   only the frames listed in the view list
    annotations/job_<id>.xml                              annotation of the views that are kept
    job_frames.json                                       view list, restricted to the kept recordings
By default the kept recordings are the 15 of splits/folds.json (the thesis data). --recordings
may be the full recordings at 90 fps or a folder that holds only the annotated frames.
Personal data (participant tables) is never copied.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--job-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs-from", type=Path, default=paths.FOLDS,
                    help="folds.json whose recordings are kept; 'all' keeps every recording")
    a = ap.parse_args()

    views = json.loads(a.job_frames.read_text())
    if str(a.runs_from) == "all":
        runs = {v["task_name"].split("__")[0] for v in views}
    else:
        runs = {r for f in json.loads(a.runs_from.read_text()).values() for r in f["runs"]}
    keep = [v for v in views if v["task_name"].split("__")[0] in runs]
    missing = 0
    for v in keep:
        run, cam = v["task_name"].split("__")
        rel = Path(run) / "images" / cam[3] / cam.split("_")[1]
        dst = a.out / "recordings" / rel
        dst.mkdir(parents=True, exist_ok=True)
        for n in v["frame_names"]:
            src = a.recordings / rel / n
            if not src.exists():
                missing += 1
                continue
            if not (dst / n).exists():
                shutil.copy2(src, dst / n)
    (a.out / "annotations").mkdir(parents=True, exist_ok=True)
    n_xml = 0
    for v in keep:
        x = a.annotations / f"job_{v['job_id']}.xml"
        if x.exists():
            shutil.copy2(x, a.out / "annotations" / x.name)
            n_xml += 1
    (a.out / "job_frames.json").write_text(json.dumps(keep, indent=1))
    print(f"{len(runs)} recordings, {len(keep)} views, {n_xml} annotated views -> {a.out}")
    if missing:
        print(f"WARNING: {missing} listed frames not found in {a.recordings}")
        sys.exit(1)


if __name__ == "__main__":
    main()
