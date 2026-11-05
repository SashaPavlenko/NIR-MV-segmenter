"""
postprocess_and_export.py — Turn raw fine-tuned-model predictions
(job_<id>.npz: masks/scores/labels, written by ensemble.py / crossview_consensus.py)
into CVAT-importable per-job XML (same "CVAT 1.1" mask/track format as
to_cvat_xml.py, extended to real per-instance class labels).

Post-processing steps (each independently toggleable):
  --largest-cc / --no-largest-cc   (default: on)
      For tool classes (haptic_left/right, scalpel, screwdriver, drill —
      see categories.TOOL_CATEGORIES): a physically continuous object should
      be one connected blob per frame. Keep only the largest connected
      component of each frame's mask, drop the rest as noise.
  --dedupe-by-class / --no-dedupe-by-class   (default: on)
      The model's inference head always emits its top-10 (query, class)
      pairs above the score threshold regardless of how many real objects
      are in the scene (see video_maskformer_model.py inference_video), so
      the same class can appear more than once. At most one physical
      instance of each class exists per job (two hands, two haptics, tools),
      so if duplicates occur, keep only the highest-mean-score instance per
      class and drop the rest.

Track id = instance rank within the job; label = the real class name so
CVAT import needs no manual relabeling.

Usage:
    python src/postprocess_and_export.py \
        --raw-out outputs/x/consensus --out-dir outputs/x/cvat_xml [--jobs 390 391]
"""

import argparse
import sys
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories import CATEGORIES, CAT_ID_TO_NAME, TOOL_CATEGORIES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_cvat_xml import mask_bbox, encode_rle  # noqa: E402


def drop_small_components(mask_2d: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """Drop components that are small relative to the largest one in this frame.

    NOT "keep only the single largest": a held tool can legitimately render as
    two disconnected regions of comparable size in one frame (e.g. blade +
    handle either side of the gripping hand, or a thin midsection blending
    into a dark background) -- ground truth itself allows this (see
    cvat_to_ytvis.py's per-label mask union). Example: the scalpel of job 474
    has two near-equal components (e.g. 78/72, 93/87, 77/72 px) in most frames;
    keeping only the largest one would discard a real half of the tool.
    """
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_2d  # empty mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = areas.max()
    keep = {i + 1 for i, a in enumerate(areas) if a >= min_area_ratio * largest_area}
    return np.isin(labels_im, list(keep)).astype(mask_2d.dtype)


def apply_largest_cc(masks: np.ndarray, label_name: str, min_area_ratio: float) -> np.ndarray:
    if label_name not in TOOL_CATEGORIES:
        return masks
    T = masks.shape[0]
    out = np.zeros_like(masks)
    for t in range(T):
        if masks[t].any():
            out[t] = drop_small_components(masks[t], min_area_ratio)
    return out


def dedupe_by_class(masks: np.ndarray, scores: np.ndarray, labels: np.ndarray):
    """Keep only the highest-mean-score instance for each duplicated class."""
    keep_idx = []
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        if len(idxs) == 1:
            keep_idx.append(idxs[0])
            continue
        best = idxs[np.argmax(scores[idxs])]
        keep_idx.append(best)
    keep_idx = sorted(keep_idx)
    return masks[keep_idx], scores[keep_idx], labels[keep_idx]


def _mask_attribs(frame: int, outside: bool, **geom) -> dict:
    d = {
        "frame": str(frame),
        "outside": "1" if outside else "0",
        "occluded": "0",
        "keyframe": "1",
        "z_order": "0",
    }
    d.update({k: str(v) for k, v in geom.items()})
    return d


def build_xml(job_id: int, masks: np.ndarray, labels: np.ndarray) -> ET.Element:
    N, T, H, W = masks.shape

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    job_el = ET.SubElement(meta, "job")
    ET.SubElement(job_el, "id").text = str(job_id)
    ET.SubElement(job_el, "size").text = str(T)
    ET.SubElement(job_el, "start_frame").text = "0"
    ET.SubElement(job_el, "stop_frame").text = str(T - 1)
    labels_el = ET.SubElement(job_el, "labels")
    for name in CATEGORIES:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "type").text = "mask"

    for inst_idx in range(N):
        inst_masks = masks[inst_idx]  # (T, H, W)
        label_name = CAT_ID_TO_NAME[int(labels[inst_idx]) + 1]

        active = [t for t in range(T) if inst_masks[t].any()]
        if not active:
            continue

        track = ET.SubElement(root, "track", {
            "id": str(inst_idx),
            "label": label_name,
            "source": "auto",
        })

        last_active = active[-1]
        for t in range(T):
            mask_2d = inst_masks[t]
            bbox = mask_bbox(mask_2d)
            if bbox is not None:
                left, top, width, height = bbox
                crop = mask_2d[top:top + height, left:left + width]
                rle = encode_rle(crop)
                ET.SubElement(track, "mask", _mask_attribs(
                    t, outside=False,
                    left=left, top=top, width=width, height=height, rle=rle,
                ))
            elif t < last_active:
                ET.SubElement(track, "mask", _mask_attribs(t, outside=True))

        if last_active < T - 1:
            ET.SubElement(track, "mask", _mask_attribs(last_active + 1, outside=True))

    return root


def pretty_xml(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding=None)


def convert_job(job_id: int, npz_path: Path, out_path: Path,
                 score_thr: float, do_largest_cc: bool, do_dedupe: bool,
                 min_component_area_ratio: float):
    data = np.load(str(npz_path))
    masks, scores, labels = data["masks"], data["scores"], data["labels"]

    keep = scores > score_thr
    masks, scores, labels = masks[keep], scores[keep], labels[keep]

    if do_dedupe and len(labels):
        masks, scores, labels = dedupe_by_class(masks, scores, labels)

    if do_largest_cc:
        for i in range(len(labels)):
            label_name = CAT_ID_TO_NAME[int(labels[i]) + 1]
            masks[i] = apply_largest_cc(masks[i], label_name, min_component_area_ratio)

    N, T, H, W = masks.shape if len(masks) else (0, 0, 0, 0)
    print(f"  job {job_id:>4d}: {N} instances after post-processing "
          f"({[CAT_ID_TO_NAME[int(l)+1] for l in labels]})", flush=True)

    root = build_xml(job_id, masks, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml_str = pretty_xml(root)
    lines = xml_str.splitlines()
    if lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"           saved -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-out", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--jobs", nargs="*", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--largest-cc", dest="largest_cc", action="store_true", default=True,
                     help="drop small stray components for tool classes (default: on). "
                          "NOT 'keep only the single largest' -- see --min-component-area-ratio")
    ap.add_argument("--no-largest-cc", dest="largest_cc", action="store_false")
    ap.add_argument("--min-component-area-ratio", type=float, default=0.15,
                     help="a connected component is kept if its area >= this fraction of the "
                          "frame's largest component (default 0.15); a genuinely continuous tool "
                          "rendered as two comparable-size regions (e.g. blade+handle either side "
                          "of the gripping hand) keeps both")
    ap.add_argument("--dedupe-by-class", dest="dedupe", action="store_true", default=True)
    ap.add_argument("--no-dedupe-by-class", dest="dedupe", action="store_false")
    args = ap.parse_args()

    npz_files = sorted(args.raw_out.glob("job_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if args.jobs is not None:
        job_set = set(args.jobs)
        npz_files = [p for p in npz_files if int(p.stem.split("_")[1]) in job_set]

    print(f"Converting {len(npz_files)} job(s) "
          f"(largest_cc={args.largest_cc}, min_component_area_ratio={args.min_component_area_ratio}, "
          f"dedupe_by_class={args.dedupe})...")

    ok = 0
    for npz_path in npz_files:
        job_id = int(npz_path.stem.split("_")[1])
        out_path = args.out_dir / f"job_{job_id}.xml"
        try:
            convert_job(job_id, npz_path, out_path, args.score_thr, args.largest_cc, args.dedupe,
                        args.min_component_area_ratio)
            ok += 1
        except Exception as e:
            print(f"  [ERR] job {job_id}: {e}", file=sys.stderr)
            raise

    print(f"\nDone: {ok}/{len(npz_files)} jobs converted.")


if __name__ == "__main__":
    main()
