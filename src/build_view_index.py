"""Build the multi-view index: which jobs are simultaneous views of the same clip,
and which of their frames were captured at the same moment.

The rig was not moved between recordings, so cam1..cam4 mean the same viewpoints in
every run. Views of one run are synchronized only in software (median offset between
cameras is a few ms, occasionally up to two frame intervals), so frames are grouped
into "moments" by timestamp with a tolerance, not by frame index.

Input:
  job_frames.json    view (CVAT job) -> task_name "<run>__cam<N>_<side>" and the frame file
                     names in frame order; a name starts with the capture timestamp in us

Output (JSON):
  {"runs": {"<run>": {"views": {"cam1_left": {"job_id":..., "n_frames":...}, ...},
                      "moments": [{"t": <us>, "views": {"cam1_left": <frame_idx>, ...}}, ...]}},
   "stats": {...}}
"""
import argparse
import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

CAM_ORDER = [f"cam{c}_{s}" for c in (1, 2, 3, 4) for s in ("left", "right")]


def load_views(job_frames: Path):
    """job_id -> (run, view, {frame_idx: timestamp_us})."""
    views = {}
    for job in json.loads(job_frames.read_text()):
        run, view = job["task_name"].split("__")
        # file name is "<timestamp_us>_<side>.bmp"
        stamps = {i: int(name.split("_")[0]) for i, name in enumerate(job["frame_names"])}
        views[job["job_id"]] = (run, view, stamps)
    return views


def cluster_moments(per_view_stamps, tol_us):
    """Group (view, frame_idx) across views into moments around a reference view.

    The reference is the view with the most frames; every moment is one of its frames.
    Each other view contributes the frame closest in time, at most one per moment and
    only within tol. Cameras drift relative to each other over a clip (frame counts
    differ by up to 2 per 5 s), so a fixed frame-index offset would not hold.
    """
    ref_view = max(per_view_stamps, key=lambda v: len(per_view_stamps[v]))
    ref = sorted(per_view_stamps[ref_view].items(), key=lambda kv: kv[1])
    ref_t = [t for _, t in ref]
    moments = [{"t": t, "views": {ref_view: idx}} for idx, t in ref]

    for view, stamps in per_view_stamps.items():
        if view == ref_view:
            continue
        best = {}  # moment index -> (|dt|, frame index)
        for idx, t in stamps.items():
            j = bisect.bisect_left(ref_t, t)
            for cand in (j - 1, j):
                if not 0 <= cand < len(ref_t):
                    continue
                dt = abs(ref_t[cand] - t)
                if dt <= tol_us and (cand not in best or dt < best[cand][0]):
                    best[cand] = (dt, idx)
        for cand, (_, idx) in best.items():
            moments[cand]["views"][view] = idx
    return moments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-frames", type=Path, default=paths.JOB_FRAMES)
    ap.add_argument("--tolerance-ms", type=float, default=20.0,
                    help="max spread within one moment; default is half a frame interval at 30fps")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tol_us = int(args.tolerance_ms * 1000)
    views = load_views(args.job_frames)

    by_run = defaultdict(dict)
    for jid, (run, view, stamps) in views.items():
        by_run[run][view] = (jid, stamps)

    out = {"tolerance_us": tol_us, "runs": {}}
    spreads, full_counts = [], []
    for run in sorted(by_run):
        per_view_stamps = {v: s for v, (_, s) in by_run[run].items()}
        moments = cluster_moments(per_view_stamps, tol_us)
        # spread inside each moment, for a sanity report
        for m in moments:
            ts = [per_view_stamps[v][i] for v, i in m["views"].items()]
            spreads.append((max(ts) - min(ts)) / 1000.0)
        n_views = len(per_view_stamps)
        full = sum(1 for m in moments if len(m["views"]) == n_views)
        full_counts.append((run, full, len(moments), n_views))
        out["runs"][run] = {
            "views": {
                v: {"job_id": jid, "n_frames": len(s)}
                for v, (jid, s) in sorted(by_run[run].items())
            },
            "moments": moments,
        }

    out["stats"] = {
        "runs": len(out["runs"]),
        "views": len(views),
        "moments_total": sum(len(r["moments"]) for r in out["runs"].values()),
        "moment_spread_ms": {
            "median": round(statistics.median(spreads), 2),
            "p90": round(sorted(spreads)[int(0.9 * len(spreads))], 2),
            "max": round(max(spreads), 2),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"runs {out['stats']['runs']}  views {out['stats']['views']}  "
          f"moments {out['stats']['moments_total']}")
    print(f"spread inside a moment (ms): {out['stats']['moment_spread_ms']}")
    print("\nrun                       moments  complete(all views)  views")
    for run, full, total, n in full_counts:
        print(f"{run:24s} {total:7d}  {full:19d}  {n:5d}")


if __name__ == "__main__":
    main()
