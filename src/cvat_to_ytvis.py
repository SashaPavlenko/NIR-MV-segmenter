"""
cvat_to_ytvis.py — Convert CVAT-exported mask annotations into a YTVIS-format
JSON that HOISTFormer's dataset loader (mask2former_video/data_video/datasets/hoist.py)
can consume directly.

Mapping notes:
  - One CVAT job (= one camera/side for one run) = one YTVIS "video".
  - Categories are the 8 CVAT labels, ids 1..8, kept in a fixed order so the
    inference->CVAT export round-trips without manual remapping.
  - At most one instance per category per job (physical constraint: <=2 hands,
    <=2 haptics, one active tool), so category identity alone is enough to
    track an instance across the whole clip — no CVAT track IDs exist in the
    export anyway.
  - hand_mask / contact_mask are HOISTFormer-specific auxiliary supervision
    fields required by the loader for every instance. We are not using
    HOIST's original hand/held-object formulation, so
    these are filled with the same mask as the main segmentation (the
    corresponding loss terms become redundant duplicates of the mask loss,
    but the pipeline shape is preserved).
  - CVAT XML <image id="..."> is the ORIGINAL (unfiltered) frame number, not
    a 0-based sequential index — it must be resolved through job_frames.json's
    frame_ids list to find the actual file. (Verified: sequential-index
    assumption produces silent mismatches for frame_filter step=3 jobs.)

Annotation export layout: one XML per job, data/annotations/job_<id>.xml (job id in the
file name and in <meta><job><id>). task_name per job_id comes from job_frames.json.

Usage:
    python src/cvat_to_ytvis.py \
        --job-frames data/job_frames.json \
        --cvat-xml-dir data/annotations \
        --data-root data/recordings \
        --out outputs/train_data/all.json
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_cvat_xml import decode_rle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories import CATEGORIES, CAT_NAME_TO_ID  # noqa: E402
from clip_metadata import activity_id_of, camera_pos_id_of  # noqa: E402

TASK_NAME_RE = re.compile(r"^(.+)__cam(\d+)_(left|right)$")
JOB_XML_NAME_RE = re.compile(r"^job_(\d+)\.xml$")


def resolve_image_dir(data_root: Path, task_name: str) -> Path:
    m = TASK_NAME_RE.match(task_name)
    if not m:
        raise ValueError(f"Unexpected task_name format: {task_name!r}")
    run_dir, cam, side = m.groups()
    return data_root / run_dir / "images" / cam / side


def encode_mask_for_json(bitmask: np.ndarray) -> dict:
    """COCO-compressed RLE with counts decoded to str for JSON serialization."""
    rle = mask_util.encode(np.asfortranarray(bitmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def convert_job(job: dict, xml_path: Path, data_root: Path, next_ann_id: int):
    """Returns (video_dict, list_of_annotation_dicts, updated_next_ann_id)."""
    task_name = job["task_name"]
    frame_ids = job["frame_ids"]
    frame_names = job["frame_names"]
    assert len(frame_ids) == len(frame_names)
    fid_to_pos = {fid: pos for pos, fid in enumerate(frame_ids)}

    image_dir = resolve_image_dir(data_root, task_name)
    length = len(frame_names)

    video = {
        "id": job["job_id"],
        "file_names": [str((image_dir / name)) for name in frame_names],
        "height": None,  # filled in below from the first parsed <image>
        "width": None,
        "length": length,
        # Per-clip conditioning metadata for the ablation configs (see
        # clip_metadata.py); consumed by
        # HOISTFormer/mask2former_video/data_video/datasets/hoist.py.
        "activity_id": activity_id_of(task_name),
        "camera_pos_id": camera_pos_id_of(task_name),
    }

    # per-category running annotation, indexed by category name
    per_cat = {}  # name -> dict(segmentations, bboxes, areas)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    for img_el in root.findall("image"):
        fid = int(img_el.get("id"))
        pos = fid_to_pos.get(fid)
        if pos is None:
            raise ValueError(
                f"{xml_path}: frame id {fid} not found in job_frames.json "
                f"frame_ids for job {job['job_id']}"
            )
        if frame_names[pos] != img_el.get("name"):
            raise ValueError(
                f"{xml_path}: frame name mismatch at id={fid}: "
                f"json={frame_names[pos]!r} xml={img_el.get('name')!r}"
            )
        w, h = int(img_el.get("width")), int(img_el.get("height"))
        if video["width"] is None:
            video["width"], video["height"] = w, h
        elif (video["width"], video["height"]) != (w, h):
            raise ValueError(f"{xml_path}: inconsistent image size at frame {fid}")

        # A label can appear as more than one <mask> in the same image (e.g. an
        # occluded object split into disconnected regions) — union them into a
        # single per-label mask for this frame instead of keeping only the last.
        combined_by_label = {}
        for mask_el in img_el.findall("mask"):
            label = mask_el.get("label")
            if label not in CAT_NAME_TO_ID:
                raise ValueError(f"{xml_path}: unknown label {label!r}")
            left = int(mask_el.get("left"))
            top = int(mask_el.get("top"))
            mw = int(mask_el.get("width"))
            mh = int(mask_el.get("height"))
            crop = decode_rle(mask_el.get("rle"), mw, mh)

            full = combined_by_label.get(label)
            if full is None:
                full = np.zeros((h, w), dtype=np.uint8)
                combined_by_label[label] = full
            full[top:top + mh, left:left + mw] |= (crop > 0).astype(np.uint8)

        for label, full in combined_by_label.items():
            entry = per_cat.setdefault(
                label,
                {
                    "segmentations": [None] * length,
                    "bboxes": [None] * length,
                    "areas": [None] * length,
                },
            )
            area = int(full.sum())
            if area == 0:
                continue
            rle = encode_mask_for_json(full)
            ys, xs = np.where(full)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            entry["segmentations"][pos] = rle
            entry["bboxes"][pos] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
            entry["areas"][pos] = area

    annotations = []
    for label, entry in per_cat.items():
        segm = entry["segmentations"]
        hand_mask = [None if s is None else s for s in segm]  # placeholder = same mask
        contact_mask = [None if s is None else s for s in segm]
        ann = {
            "id": next_ann_id,
            "video_id": job["job_id"],
            "category_id": CAT_NAME_TO_ID[label],
            "iscrowd": 0,
            "height": video["height"],
            "width": video["width"],
            "length": length,
            "segmentations": segm,
            "bboxes": entry["bboxes"],
            "areas": entry["areas"],
            "hand_masks": hand_mask,
            "contact_masks": contact_mask,
        }
        annotations.append(ann)
        next_ann_id += 1

    return video, annotations, next_ann_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-frames", required=True, type=Path)
    ap.add_argument("--cvat-xml-dir", required=True, type=Path, nargs="+",
                     help="one or more flat directories of job_<id>.xml exports, e.g. "
                          "data/annotations data/annotations_v2 (a job present in several "
                          "dirs is taken from the LAST one listed -- newest export wins)")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--jobs",
        nargs="*",
        type=int,
        default=None,
        help="restrict to these job_ids (default: all jobs that have a CVAT export)",
    )
    args = ap.parse_args()

    all_jobs = {j["job_id"]: j for j in json.loads(args.job_frames.read_text())}

    videos = []
    annotations = []
    next_ann_id = 1
    n_converted = 0
    xml_by_job = {}
    for xml_dir in args.cvat_xml_dir:
        for xml_path in sorted(xml_dir.glob("job_*.xml")):
            m = JOB_XML_NAME_RE.match(xml_path.name)
            if not m:
                raise ValueError(f"Unexpected annotation filename: {xml_path.name!r}")
            xml_by_job[int(m.group(1))] = xml_path  # later dirs override earlier ones

    for job_id, xml_path in sorted(xml_by_job.items()):
        xml_job_id = int(ET.parse(xml_path).getroot().find(".//job/id").text)
        if xml_job_id != job_id:
            raise ValueError(
                f"{xml_path}: filename job id {job_id} != <meta><job><id> {xml_job_id}"
            )
        if args.jobs is not None and job_id not in args.jobs:
            continue
        job = all_jobs.get(job_id)
        if job is None:
            raise ValueError(f"{xml_path}: job_id {job_id} not found in job_frames.json")
        video, anns, next_ann_id = convert_job(job, xml_path, args.data_root, next_ann_id)
        videos.append(video)
        annotations.extend(anns)
        n_converted += 1

    categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(CATEGORIES)
    ]
    out = {"videos": videos, "annotations": annotations, "categories": categories}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"Converted {n_converted} job(s) -> {len(annotations)} instance annotations")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
