"""Disparity channel for every frame of every view in a view list (input of the depth model).

    python src/precompute_depth.py --job-frames data/job_frames.json --recordings data/recordings \
        --out outputs/x/depth_frames

For each recording, device and frame: rectify both eyes with the device grids
(calibration/leap_grids), run SGM, map the disparity back into the raw frame of each eye.
Output: one 3-channel PNG per frame (BGR = disparity * 2, NIR, NIR; 0 = no match) in the same
folder layout as the recordings: <out>/<run>/images/<device>/<side>/<timestamp>_<side>.png
Only frames listed in --job-frames are processed. A frame needs the image of the other eye
with the same timestamp.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from leap_geometry import eye, right_from_left, sgm  # noqa: E402

ARGS = None


def process(task):
    run, dev, stamps = task
    el, er = eye(dev, "left"), eye(dev, "right")
    src = ARGS.recordings / run / "images" / str(dev)
    dst = ARGS.out / run / "images" / str(dev)
    out_l, out_r = dst / "left", dst / "right"
    out_l.mkdir(parents=True, exist_ok=True)
    out_r.mkdir(parents=True, exist_ok=True)
    done, missing, cover = 0, 0, []
    for ts in sorted(stamps):
        if (out_l / f"{ts}_left.png").exists() and (out_r / f"{ts}_right.png").exists():
            continue
        L = cv2.imread(str(src / "left" / f"{ts}_left.bmp"), 0)
        R = cv2.imread(str(src / "right" / f"{ts}_right.bmp"), 0)
        if L is None or R is None:
            missing += 1
            continue
        dL = sgm(el.to_rect(L), er.to_rect(R))
        dR = right_from_left(dL)
        for img, d, e, od, side in ((L, dL, el, out_l, "left"), (R, dR, er, out_r, "right")):
            draw = e.to_raw(d, cv2.INTER_NEAREST)
            ch = np.clip(np.round(draw * 2), 0, 255).astype(np.uint8)
            cv2.imwrite(str(od / f"{ts}_{side}.png"), np.dstack([ch, img, img]))   # BGR: depth, nir, nir
            cover.append(float((draw > 0).mean()))
        done += 1
    return run, dev, done, missing, float(np.mean(cover)) if cover else 0.0


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-frames", type=Path, default=paths.JOB_FRAMES)
    ap.add_argument("--recordings", type=Path, default=paths.RECORDINGS_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, nargs="*", default=None, help="default: every view")
    ap.add_argument("--workers", type=int, default=16)
    ARGS = ap.parse_args()

    # (run, device) -> timestamps; both eyes of a device are processed together
    stamps = defaultdict(set)
    for job in json.loads(ARGS.job_frames.read_text()):
        if ARGS.jobs and job["job_id"] not in ARGS.jobs:
            continue
        run, cam = job["task_name"].split("__")
        dev = int(cam[3])
        for n in job["frame_names"]:
            stamps[(run, dev)].add(n.split("_")[0])
    tasks = [(run, dev, s) for (run, dev), s in sorted(stamps.items())]
    with Pool(ARGS.workers) as pool:
        for run, dev, n, missing, cover in pool.imap_unordered(process, tasks):
            msg = f"{run} device {dev}: {n} new frame pairs, disparity on {cover:.0%} of pixels"
            print(msg + (f", {missing} frames without a stereo partner" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
