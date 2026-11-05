"""Cross-view consensus over single-view predictions (variant A, no calibration).

Four views of the same clip see the same physical scene: one tool, the same two hands,
the same two haptics. The single-view model does not know this, so it disagrees with
itself across views -- on the v4 combo predictions, 10 of 15 runs carry a different
dominant tool class in different cameras, although each run contains exactly one tool.

This script reads the per-view raw predictions, aligns their frames through
the view index of build_view_index.py (timestamp matching, no geometry involved) and lets the views vote.
Rules are independent and individually switchable:

  R1 --tool-vote         one tool class per clip. OFF by default, and kept only as an
                         ablation: the premise is false. The ground truth shows a scalpel
                         lying in the scene throughout the drill recordings (11 annotated
                         views carry drill and scalpel together), so forcing a single tool
                         deletes correct masks -- measured: scalpel recall 1.00 -> 0.00 on
                         five holdout views. A recording name names the active tool, not
                         the only one present.
  R2 --class-vote        a class claimed by too few views is a ghost and is dropped in
                         the views that claim it.
  R3 --frame-vote        per moment: a detection isolated to a single view is dropped;
                         a view missing a class the others agree on is marked as a gap.
  R4 --temporal-fill     fill those gaps from the nearest frame of the same view
                         (within --max-fill-gap frames); never invents a mask in a view
                         where the class is absent throughout.

Output has the same .npz layout as the input, so postprocess_and_export.py and
visualize_predictions.py run on it unchanged.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories import CATEGORIES  # noqa: E402

NAMES = [c["name"] if isinstance(c, dict) else c for c in CATEGORIES]
TOOL_CLASSES = [NAMES.index(n) for n in ("scalpel", "screwdriver", "drill")]


def read_view(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    masks, labels, scores = z["masks"], z["labels"], z["scores"]
    area = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2)
    present = area > 0
    return labels, scores, present, area, masks.shape[1]


def collect_run(run_info, raw_dir):
    """view -> dict(labels, scores, present, n_frames); views without predictions are skipped."""
    out = {}
    for view, info in run_info["views"].items():
        p = raw_dir / f"job_{info['job_id']}.npz"
        if not p.exists():
            continue
        labels, scores, present, area, nf = read_view(p)
        out[view] = {"job_id": info["job_id"], "labels": labels, "scores": scores,
                     "present": present, "area": area, "n_frames": nf}
    return out


def class_evidence(views, kind):
    """class -> {view: evidence}. Evidence weighted by mask area, by frame count, or both.

    Area matters because the failure this rule has to fix is a second tool class covering
    a few dozen pixels next to the real tool: it is present in as many frames as the real
    one (so a frame count cannot separate them) and scored just as confidently, but it is
    an order of magnitude smaller.
    """
    ev = defaultdict(dict)
    for view, v in views.items():
        for i, lab in enumerate(v["labels"]):
            lab = int(lab)
            frames = float(v["present"][i].sum())
            area = float(v["area"][i].sum())
            w = {"frames": frames, "area": area, "ballot": area}[kind]
            ev[lab][view] = ev[lab].get(view, 0.0) + w * float(v["scores"][i])
    return ev


def decide(views, index_run, args):
    """Return per-view decisions: classes to drop, frames to suppress, gaps to fill."""
    n_views = len(views)
    ev = class_evidence(views, args.tool_evidence)
    drop = defaultdict(set)        # view -> {class}
    suppress = defaultdict(set)    # view -> {(class, frame)}
    fill = defaultdict(set)        # view -> {(class, frame)}
    report = {"n_views": n_views}

    # ---- R1: a single tool class for the whole clip -------------------------------
    if args.tool_vote:
        if args.tool_evidence == "ballot":
            # one vote per view, so a single view with a large blob cannot outweigh the rest
            ballots = defaultdict(int)
            for view in views:
                per_class = {c: ev.get(c, {}).get(view, 0.0) for c in TOOL_CLASSES}
                if max(per_class.values()) > 0:
                    ballots[max(per_class, key=per_class.get)] += 1
            totals = {c: float(ballots.get(c, 0)) for c in TOOL_CLASSES}
        else:
            totals = {c: sum(ev.get(c, {}).values()) for c in TOOL_CLASSES}
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        winner, best = ranked[0]
        runner, second = ranked[1]
        if best > 0:
            margin = (best - second) / best
            report["tool_vote"] = {
                "winner": NAMES[winner],
                "margin": round(margin, 3),
                "evidence": {NAMES[c]: round(v, 1) for c, v in totals.items()},
                "views_supporting": {NAMES[c]: sorted(ev.get(c, {})) for c in TOOL_CLASSES},
            }
            if margin >= args.tool_margin:
                for c in TOOL_CLASSES:
                    if c != winner:
                        for view in ev.get(c, {}):
                            drop[view].add(c)
            else:
                report["tool_vote"]["applied"] = False

    # ---- R2: a class claimed by too few views is a ghost ---------------------------
    if args.class_vote:
        votes = {}
        for c, per_view in ev.items():
            # a view "claims" the class only if it holds it for a real stretch of frames
            claiming = [
                view for view, _ in per_view.items()
                if claim_fraction(views[view], c) >= args.min_frame_frac
            ]
            votes[NAMES[c]] = len(claiming)
            if n_views >= args.min_views_for_vote and len(claiming) / n_views < args.min_view_support:
                for view in per_view:
                    drop[view].add(c)
        report["class_votes"] = votes

    # ---- R3 / R4: per-moment agreement ---------------------------------------------
    if args.frame_vote or args.temporal_fill:
        stats = {"suppressed": 0, "gaps": 0}
        for moment in index_run["moments"]:
            participants = {v: f for v, f in moment["views"].items() if v in views}
            if len(participants) < args.min_views_for_vote:
                continue
            for c in set(int(l) for v in views.values() for l in v["labels"]):
                holders, missing = [], []
                for view, f in participants.items():
                    if c in drop[view]:
                        continue
                    v = views[view]
                    idx = [i for i, l in enumerate(v["labels"]) if int(l) == c]
                    if not idx:
                        continue   # class absent in this view entirely -> nothing to fill
                    if f < v["n_frames"] and any(v["present"][i][f] for i in idx):
                        holders.append(view)
                    else:
                        missing.append((view, f))
                n_part = len(holders) + len(missing)
                if n_part < args.min_views_for_vote:
                    continue
                share = len(holders) / n_part
                if args.frame_vote and len(holders) == 1 and n_part >= args.min_views_for_vote \
                        and share <= args.isolated_share:
                    suppress[holders[0]].add((c, participants[holders[0]]))
                    stats["suppressed"] += 1
                elif share >= args.fill_share:
                    for view, f in missing:
                        fill[view].add((c, f))
                        stats["gaps"] += 1
        report["frame_vote"] = stats

    return drop, suppress, fill, report


def drop_all(drop):
    return set().union(*drop.values()) if drop else set()


def claim_fraction(view, c):
    idx = [i for i, l in enumerate(view["labels"]) if int(l) == c]
    if not idx:
        return 0.0
    return float(np.any(view["present"][idx], axis=0).sum()) / max(view["n_frames"], 1)


def apply_to_view(npz_in, npz_out, drop_classes, suppress, fill, args):
    z = np.load(npz_in, allow_pickle=True)
    masks, labels, scores = z["masks"].copy(), z["labels"], z["scores"]
    keep = [i for i, l in enumerate(labels) if int(l) not in drop_classes]
    changed = {"dropped_instances": len(labels) - len(keep), "suppressed_frames": 0,
               "filled_frames": 0}

    by_class = defaultdict(list)
    for i in keep:
        by_class[int(labels[i])].append(i)

    for c, f in suppress:
        for i in by_class.get(c, []):
            if masks[i, f].any():
                masks[i, f] = 0
                changed["suppressed_frames"] += 1

    if args.temporal_fill:
        for c, f in sorted(fill):
            for i in by_class.get(c, []):
                if masks[i, f].any():
                    continue
                src = nearest_filled(masks[i], f, args.max_fill_gap)
                if src is not None:
                    masks[i, f] = masks[i, src]
                    changed["filled_frames"] += 1

    masks, labels, scores = masks[keep], labels[keep], scores[keep]
    np.savez_compressed(npz_out, masks=masks, labels=labels, scores=scores)
    return changed


def nearest_filled(mask_seq, f, max_gap):
    for d in range(1, max_gap + 1):
        for cand in (f - d, f + d):
            if 0 <= cand < mask_seq.shape[0] and mask_seq[cand].any():
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True, help="written by build_view_index.py")
    ap.add_argument("--raw-in", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="default: every run with predictions")
    ap.add_argument("--report", type=Path, default=None)

    ap.add_argument("--tool-vote", dest="tool_vote", action="store_true", default=False,
                    help="ablation only: several tools really do coexist in a recording")
    ap.add_argument("--no-tool-vote", dest="tool_vote", action="store_false")
    ap.add_argument("--tool-evidence", choices=("ballot", "area", "frames"), default="ballot",
                    help="ballot: each view votes for its largest tool, majority wins (default); "
                         "area/frames: pooled evidence across views")
    ap.add_argument("--tool-margin", type=float, default=0.15,
                    help="relative lead the winning tool needs; below this the vote is not applied")
    ap.add_argument("--class-vote", dest="class_vote", action="store_true", default=True)
    ap.add_argument("--no-class-vote", dest="class_vote", action="store_false")
    ap.add_argument("--min-view-support", type=float, default=0.3,
                    help="fraction of views that must claim a class for it to survive")
    ap.add_argument("--min-frame-frac", type=float, default=0.2,
                    help="fraction of its frames a view must hold a class to count as claiming it")
    ap.add_argument("--frame-vote", dest="frame_vote", action="store_true", default=True)
    ap.add_argument("--no-frame-vote", dest="frame_vote", action="store_false")
    ap.add_argument("--isolated-share", type=float, default=0.25,
                    help="a detection present in at most this share of views at a moment is dropped")
    ap.add_argument("--temporal-fill", dest="temporal_fill", action="store_true", default=True)
    ap.add_argument("--no-temporal-fill", dest="temporal_fill", action="store_false")
    ap.add_argument("--fill-share", type=float, default=0.6,
                    help="share of views that must hold a class at a moment to call a miss a gap")
    ap.add_argument("--max-fill-gap", type=int, default=3)
    ap.add_argument("--min-views-for-vote", type=int, default=3,
                    help="runs with fewer predicted views are copied through untouched")
    args = ap.parse_args()

    index = json.loads(args.index.read_text())
    args.raw_out.mkdir(parents=True, exist_ok=True)
    full_report = {"args": {k: str(v) for k, v in vars(args).items()}, "runs": {}}

    runs = args.runs or sorted(index["runs"])
    for run in runs:
        info = index["runs"][run]
        views = collect_run(info, args.raw_in)
        if not views:
            continue
        if len(views) < args.min_views_for_vote:
            for view, v in views.items():
                src = args.raw_in / f"job_{v['job_id']}.npz"
                (args.raw_out / src.name).write_bytes(src.read_bytes())
            full_report["runs"][run] = {"skipped": f"only {len(views)} views with predictions"}
            print(f"{run:24s} {len(views)} views -> copied through (too few to vote)")
            continue

        drop, suppress, fill, report = decide(views, info, args)
        per_view = {}
        for view, v in views.items():
            src = args.raw_in / f"job_{v['job_id']}.npz"
            changed = apply_to_view(src, args.raw_out / src.name, drop[view],
                                    suppress[view], fill[view], args)
            per_view[view] = {"job_id": v["job_id"], **changed}
        report["views"] = per_view
        full_report["runs"][run] = report

        tv = report.get("tool_vote", {})
        dropped = sum(p["dropped_instances"] for p in per_view.values())
        print(f"{run:24s} {len(views)} views  tool={tv.get('winner','-'):11s} "
              f"margin={tv.get('margin', 0):.2f}  dropped_inst={dropped:3d}  "
              f"suppressed={sum(p['suppressed_frames'] for p in per_view.values()):4d}  "
              f"filled={sum(p['filled_frames'] for p in per_view.values()):4d}")

    out = args.report or (args.raw_out.parent / f"{args.raw_out.name}_report.json")
    out.write_text(json.dumps(full_report, indent=1))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
