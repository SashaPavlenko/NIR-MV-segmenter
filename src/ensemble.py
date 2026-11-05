"""Combine the soft predictions of the NIR model and the NIR + disparity model into binary masks.

    python src/ensemble.py --soft outputs/run/soft_nir outputs/run/soft_depth --out outputs/run/ensemble

For every view (job_<id>.npz present in all --soft folders):
  1. overlapping windows: every frame averages the probabilities of all windows (stride 8)
     that detect a class (score > --thr),
  2. the models are averaged (a class detected by only one model keeps that model's map),
  3. pixel > --cut, small components of tool and haptic classes are dropped.
Output: <out>/job_<id>.npz in the raw format of crossview_consensus.py.
With one --soft folder this is the single-model pipeline with overlapping windows.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import masks as M  # noqa: E402

ARGS = None


def one(name):
    out = ARGS.out / name
    if out.exists() and not ARGS.force:
        return name, "exists"
    p, q = M.assemble(M.load_soft(ARGS.soft[0] / name), mode=ARGS.mode, thr=ARGS.thr)
    for k, d in enumerate(ARGS.soft[1:]):
        p2, q2 = M.assemble(M.load_soft(d / name), mode=ARGS.mode, thr=ARGS.thr)
        p, q = M.ensemble(p, q, p2, q2, k + 1)
    M.to_raw_npz(M.binarize(p, q, ARGS.cut), out)
    return name, "ok"


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soft", type=Path, nargs="+", required=True, help="folders written by infer_soft.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("overlap", "nonoverlap"), default="overlap")
    ap.add_argument("--thr", type=float, default=0.5, help="class score threshold per window")
    ap.add_argument("--cut", type=float, default=0.5, help="pixel probability threshold")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in ARGS.soft[0].glob("job_*.npz"))
    names = [n for n in names if all((d / n).exists() for d in ARGS.soft[1:])]
    if not names:
        sys.exit(f"no common job_*.npz in {[str(d) for d in ARGS.soft]}")
    with Pool(ARGS.workers) as pool:
        for name, st in pool.imap_unordered(one, names):
            print(f"{name}: {st}", flush=True)
    print(f"{len(names)} views -> {ARGS.out}")


if __name__ == "__main__":
    main()
