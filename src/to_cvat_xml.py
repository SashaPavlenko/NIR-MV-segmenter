"""
to_cvat_xml.py — Convert raw HOISTFormer predictions to CVAT 1.1 track XML.

For each raw_out/job_<id>.npz produces cvat_xml/job_<id>.xml importable via
CVAT REST API or UI (format: "CVAT 1.1").

Instance → track mapping:
  - All instances get temporary label "object" (all model labels are class 0;
    the user reassigns hand_left / hand_right / tool labels inside CVAT).
  - Track id = instance index (0-based).

Frame numbering:
  - Uses local job frame index 0..T-1, which matches CVAT's per-job numbering
    and the order produced by prepare_frames.py.

RLE encoding:
  - Crop mask to its tight bounding box (left, top, width, height).
  - Flatten row-major (C order) within the crop — this is CVAT's actual
    convention (verified against real CVAT-exported masks; an earlier
    version of this file used column-major and produced corrupted masks).
  - Emit alternating counts of 0-pixels then 1-pixels, starting with 0-count
    (may be 0 if the first pixel is foreground).

Outside handling:
  - Only frames with a non-empty mask get outside="0" + geometry.
  - One outside="1" keyframe is appended after the last active frame
    (unless it is the last frame of the job), so CVAT knows the track ended.
  - Gaps inside the track (empty mask frames between two active frames) each
    get an explicit outside="1" keyframe so CVAT doesn't interpolate across them.

Usage:
  python src/to_cvat_xml.py --raw-out <dir of job_<id>.npz> --out-dir <dir> [--jobs 412 413 ...]

The functions mask_bbox / encode_rle / decode_rle are used by postprocess_and_export.py and
cvat_to_ytvis.py. The command line itself is the class-agnostic export of the first
pre-annotation round (every instance labelled "object").
"""

import argparse
import numpy as np
import sys
import os
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET


# ── geometry helpers ──────────────────────────────────────────────────────────

def mask_bbox(mask_2d: np.ndarray):
    """Tight bounding box of a binary mask.
    Returns (left, top, width, height) or None if mask is all-zero."""
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.where(rows)[0][[0, -1]].reshape(2)[0]), \
                 int(np.where(rows)[0][[0, -1]].reshape(2)[1])
    cmin, cmax = int(np.where(cols)[0][[0, -1]].reshape(2)[0]), \
                 int(np.where(cols)[0][[0, -1]].reshape(2)[1])
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1


def encode_rle(crop: np.ndarray) -> str:
    """CVAT RLE: row-major alternating 0-count / 1-count, starting with 0."""
    flat = crop.flatten(order='C')
    counts = []
    current = 0
    expected = 0
    for px in flat:
        val = int(px > 0)
        if val == expected:
            current += 1
        else:
            counts.append(current)
            current = 1
            expected = 1 - expected
    counts.append(current)
    return ','.join(map(str, counts))


def decode_rle(rle_str: str, width: int, height: int) -> np.ndarray:
    """Inverse of encode_rle — used in the round-trip test."""
    counts = list(map(int, rle_str.split(',')))
    flat = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    val = 0
    for count in counts:
        flat[idx:idx + count] = val
        idx += count
        val = 1 - val
    return flat.reshape((height, width), order='C')


# ── XML builder ───────────────────────────────────────────────────────────────

def _mask_attribs(frame: int, outside: bool, **geom) -> dict:
    d = {
        'frame':    str(frame),
        'outside':  '1' if outside else '0',
        'occluded': '0',
        'keyframe': '1',
        'z_order':  '0',
    }
    d.update({k: str(v) for k, v in geom.items()})
    return d


def build_xml(job_id: int, masks: np.ndarray,
              scores: np.ndarray, labels: np.ndarray) -> ET.Element:
    N, T, H, W = masks.shape

    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'

    meta = ET.SubElement(root, 'meta')
    job_el = ET.SubElement(meta, 'job')
    ET.SubElement(job_el, 'id').text = str(job_id)
    ET.SubElement(job_el, 'size').text = str(T)
    ET.SubElement(job_el, 'start_frame').text = '0'
    ET.SubElement(job_el, 'stop_frame').text = str(T - 1)

    for inst_idx in range(N):
        inst_masks = masks[inst_idx]  # (T, H, W)

        # Find active (non-empty) frames
        active = [t for t in range(T) if inst_masks[t].any()]
        if not active:
            continue

        track = ET.SubElement(root, 'track', {
            'id':     str(inst_idx),
            'label':  'object',
            'source': 'auto',
        })

        last_active = active[-1]
        prev_was_active = False

        for t in range(T):
            mask_2d = inst_masks[t]
            bbox = mask_bbox(mask_2d)

            if bbox is not None:
                left, top, width, height = bbox
                crop = mask_2d[top:top + height, left:left + width]
                rle = encode_rle(crop)
                ET.SubElement(track, 'mask', _mask_attribs(
                    t, outside=False,
                    left=left, top=top, width=width, height=height, rle=rle,
                ))
                prev_was_active = True
            else:
                # Gap inside track: explicit outside keyframe
                if t < last_active:
                    ET.SubElement(track, 'mask', _mask_attribs(t, outside=True))
                prev_was_active = False

        # Terminate track after last active frame
        if last_active < T - 1:
            ET.SubElement(track, 'mask', _mask_attribs(last_active + 1, outside=True))

    return root


def pretty_xml(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding='unicode')
    return minidom.parseString(raw).toprettyxml(indent='  ', encoding=None)


# ── main ──────────────────────────────────────────────────────────────────────

def convert_job(job_id: int, npz_path: Path, out_path: Path):
    data   = np.load(str(npz_path))
    masks  = data['masks']   # (N, T, H, W) uint8
    scores = data['scores']
    labels = data['labels']

    N, T, H, W = masks.shape
    print(f'  job {job_id:>4d}: {N} instances, {T} frames, {H}×{W}px', flush=True)

    root = build_xml(job_id, masks, scores, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml_str = pretty_xml(root)
    # minidom adds its own declaration; strip it and write utf-8
    lines = xml_str.splitlines()
    if lines[0].startswith('<?xml'):
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'           saved → {out_path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-out', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--jobs',    nargs='*', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    npz_files = sorted(args.raw_out.glob('job_*.npz'),
                       key=lambda p: int(p.stem.split('_')[1]))

    if args.jobs:
        job_set  = set(args.jobs)
        npz_files = [p for p in npz_files
                     if int(p.stem.split('_')[1]) in job_set]

    print(f'Converting {len(npz_files)} job(s)...')

    ok = 0
    for npz_path in npz_files:
        job_id   = int(npz_path.stem.split('_')[1])
        out_path = args.out_dir / f'job_{job_id}.xml'

        if out_path.exists() and not args.overwrite:
            print(f'  [SKIP] job {job_id}: exists (--overwrite to redo)')
            ok += 1
            continue

        try:
            convert_job(job_id, npz_path, out_path)
            ok += 1
        except Exception as e:
            print(f'  [ERR] job {job_id}: {e}', file=sys.stderr)

    print(f'\nDone: {ok}/{len(npz_files)} jobs converted.')


if __name__ == '__main__':
    main()
