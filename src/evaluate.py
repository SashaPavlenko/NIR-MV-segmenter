"""Score predicted masks against the CVAT annotation, with the metrics of the thesis.

    python src/evaluate.py --pred outputs/reproduce/consensus --out outputs/reproduce/eval

For every annotated view (job_<id>.xml in --annotations) that has a prediction in --pred:
  * per class present in the ground truth of the view, plus tool_any (union of the three tools):
    mean IoU over the frames where ground truth or prediction is non-empty,
    frame-level presence precision / recall / F1 over all frames;
  * aggregate = mean over views. mIoU (7 classes) = mean of the seven class mIoUs
    (hand_left, hand_right, haptic_left, haptic_right, scalpel, screwdriver, drill).
Output: scores.json (per view), summary.json and summary.md (aggregate table).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from categories import CATEGORIES  # noqa: E402
from masks import FRAME_SIZE, from_raw_npz  # noqa: E402

TOOLS = ["scalpel", "screwdriver", "drill"]
LABELS = CATEGORIES[:7] + ["tool_any"]
ARGS = None


def decode_rle(rle, w, h):
    counts = [int(float(x)) for x in rle.split(",")]
    flat = np.zeros(w * h, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape(h, w)


def load_gt(job, xml_path):
    """dict class -> (T, H, W) bool from a CVAT for images 1.1 export of one job."""
    pos_of = {fid: i for i, fid in enumerate(job["frame_ids"])}
    T = len(job["frame_names"])
    gt = {c: np.zeros((T, FRAME_SIZE, FRAME_SIZE), bool) for c in CATEGORIES}
    for im in ET.parse(xml_path).getroot().findall("image"):
        t = pos_of.get(int(im.get("id")))
        if t is None:
            continue
        for m in im.findall("mask"):
            lab = m.get("label")
            if lab not in gt:
                continue
            l, tp, w, h = (int(float(m.get(k))) for k in ("left", "top", "width", "height"))
            gt[lab][t, tp:tp + h, l:l + w] |= decode_rle(m.get("rle"), w, h) > 0
    return gt


def score_view(gt, pred):
    rows = {}
    g, p = dict(gt), dict(pred)
    g["tool_any"] = np.logical_or.reduce([gt[c] for c in TOOLS])
    p["tool_any"] = np.logical_or.reduce([pred[c] for c in TOOLS])
    for lab in LABELS:
        G, P = g[lab], p[lab]
        if not G.any():
            continue
        inter = (G & P).sum(axis=(1, 2))
        union = (G | P).sum(axis=(1, 2))
        v = union > 0
        miou = float((inter[v] / union[v]).mean()) if v.any() else float("nan")
        gp, pp = G.any(axis=(1, 2)), P.any(axis=(1, 2))
        tp, fp, fn = int((gp & pp).sum()), int((~gp & pp).sum()), int((gp & ~pp).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fp + fn else float("nan"))
        rows[lab] = {"miou": miou, "precision": prec, "recall": rec, "f1": f1,
                     "n_gt_frames": int(gp.sum()), "T": len(gp)}
    return rows


def aggregate(per_view):
    """{job: {label: row}} -> {label: mean row over views}."""
    out = {}
    for lab in LABELS:
        rows = [r[lab] for r in per_view.values() if lab in r]
        if not rows:
            continue
        out[lab] = {k: float(np.nanmean([r[k] for r in rows])) for k in ("miou", "precision", "recall", "f1")}
        out[lab]["n_views"] = len(rows)
    return out


def one(job_id):
    job = ARGS.jobs_by_id[job_id]
    gt = load_gt(job, ARGS.annotations / f"job_{job_id}.xml")
    pred = from_raw_npz(ARGS.pred / f"job_{job_id}.npz", len(job["frame_names"]))
    return job_id, score_view(gt, pred)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="folder of job_<id>.npz (raw format)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=paths.ANNOTATIONS_DIR)
    ap.add_argument("--job-frames", type=Path, default=paths.JOB_FRAMES)
    ap.add_argument("--jobs", type=int, nargs="*", default=None,
                    help="default: every annotated view with a prediction")
    ap.add_argument("--workers", type=int, default=8)
    ARGS = ap.parse_args()
    ARGS.jobs_by_id = {j["job_id"]: j for j in json.loads(ARGS.job_frames.read_text())}
    jobs = ARGS.jobs or sorted(int(p.stem.split("_")[1]) for p in ARGS.annotations.glob("job_*.xml"))
    jobs = [j for j in jobs if (ARGS.pred / f"job_{j}.npz").exists()]
    if not jobs:
        sys.exit(f"no annotated view has a prediction in {ARGS.pred}")
    with Pool(ARGS.workers) as pool:
        scores = dict(pool.imap_unordered(one, jobs))
    agg = aggregate(scores)
    miou7 = float(np.mean([agg[l]["miou"] for l in CATEGORIES[:7] if l in agg]))
    ARGS.out.mkdir(parents=True, exist_ok=True)
    (ARGS.out / "scores.json").write_text(json.dumps({str(j): r for j, r in sorted(scores.items())}, indent=1))
    (ARGS.out / "summary.json").write_text(json.dumps({"n_views": len(jobs), "miou_7": miou7,
                                                       "per_class": agg}, indent=1))
    lines = [f"Views: {len(jobs)}    mIoU (7 classes): {miou7:.4f}", "",
             "| class | mIoU | presence P | presence R | presence F1 | views |",
             "|---|---|---|---|---|---|"]
    for lab in LABELS:
        if lab in agg:
            a = agg[lab]
            lines.append(f"| {lab} | {a['miou']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} "
                         f"| {a['f1']:.3f} | {a['n_views']} |")
    (ARGS.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
