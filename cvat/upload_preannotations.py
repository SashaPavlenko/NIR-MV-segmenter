#!/usr/bin/env python3
"""upload_preannotations.py -- upload model pre-annotations (CVAT 1.1 track XML) into CVAT jobs.

The XML comes from src/postprocess_and_export.py (scripts/predict.sh, step 5): one job_<id>.xml
per view, one track per class, label = the real class name (hand_left, haptic_right, scalpel, ...).
The label of each track is looked up by name in the project or task. Nothing is created
silently: if a name is missing in the project, the job is not uploaded.

Why shapes and not tracks: cvat-core has no MaskTrack class, a job with mask tracks does not
open ("An unexpected type of track mask"). The identity of an instance across frames is kept
in the `group` field (numbered from 1, because group=0 means "no group" in CVAT).

Checks BEFORE writing (a job that fails is not uploaded at all):
  * sum(rle) == width*height;
  * the bbox lies inside the frame;
  * the frame lies inside the job;
  * <meta><job><size> of the XML equals the length of the job (catches an XML of another job);
  * RLE orientation: CVAT reads the RLE row by row, rows of length width. A file written
    column-major shows diagonal stripes instead of masks. The two are told apart reliably:
    the correct reading gives about one foreground run per row, the wrong one dozens.

After writing (--verify, on by default) the annotation is read back and compared with the XML:
set of frames, number of shapes per frame, labels, groups and the area of every mask. This
catches the failure where only frame 0 arrives: it shows up as a mismatch of the frame sets,
not as an exception.

set_annotations is a PUT: it replaces the whole annotation of the job. Therefore:
  * the current state is always saved to <backup-dir>/job_<id>_<timestamp>.json;
  * shapes with source=manual and group=0 (drawn by hand) are carried over into the new
    annotation by default (--drop-manual to discard them too).

Usage:
    export CVAT_LOGIN=<user> CVAT_PASSWORD=<password>
    python3 upload_preannotations.py --xml-dir <dir> --all --dry-run
    python3 upload_preannotations.py --xml-dir <dir> --all
    python3 upload_preannotations.py --xml-dir <dir> --jobs 416 419
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType

from cvat_host import resolve_cvat_host


# ── RLE ───────────────────────────────────────────────────────────────────────

def decode_rle(rle: str, rows: int, cols: int) -> np.ndarray:
    """Decode RLE into (rows, cols). Runs alternate, the first one is background."""
    counts = [int(x) for x in rle.split(",")]
    flat = np.zeros(sum(counts), dtype=np.uint8)
    pos, val = 0, 0
    for run in counts:
        flat[pos : pos + run] = val
        pos += run
        val ^= 1
    return flat.reshape(rows, cols)


def rle_area(counts: list[int]) -> int:
    """Number of foreground pixels: runs with odd index are foreground."""
    return sum(counts[1::2])


def orientation_stats(masks: list[tuple[str, int, int]]) -> dict:
    """Compare the two readings of the RLE and tell which one the file was written in.

    An absolute threshold does not work: for a square crop, column-major is exactly a
    transpose and the "bad" image is statistically the same as the good one. So the file is
    compared with itself: every mask is decoded row-major (as CVAT reads it) and column-major,
    and the mean number of foreground runs per row is counted. For a connected mask the
    correct reading gives about one run per row, the wrong one cuts the row into pieces.
    """
    as_is = as_col = 0.0
    for rle, width, height in masks:
        row_major = decode_rle(rle, height, width)          # how CVAT reads it
        col_major = decode_rle(rle, width, height).T        # if the file was written column-major
        as_is += (np.diff(np.pad(row_major, ((0, 0), (1, 1))), axis=1) == 1).sum() / height
        as_col += (np.diff(np.pad(col_major, ((0, 0), (1, 1))), axis=1) == 1).sum() / height
    n = len(masks)
    return {"n": n, "as_is": as_is / n, "as_col_major": as_col / n}


# ── XML parsing ────────────────────────────────────────────────────────────────

class XmlError(Exception):
    pass


def parse_job_xml(xml_path: Path, frame_width: int, frame_height: int,
                  job_size: int, skip_rle_check: bool) -> list[dict]:
    """Tracks -> flat list of masks. Raises XmlError on anything invalid."""
    root = ET.parse(xml_path).getroot()

    tracks = root.findall("track")
    if not tracks:
        if root.findall("image"):
            raise XmlError(
                "the XML is in 'CVAT for images' layout (<image>), not tracks. "
                "This script expects the track XML of postprocess_and_export.py"
            )
        raise XmlError("the XML has no <track>")

    entries: list[dict] = []
    geometry: list[tuple[str, int, int]] = []

    for group_idx, track_el in enumerate(tracks, start=1):
        label = track_el.get("label")
        if not label:
            raise XmlError(f"track #{group_idx} has no label attribute")

        for m in track_el.findall("mask"):
            frame = int(m.get("frame"))
            if m.get("outside") == "1":
                continue  # object absent in this frame: no shape

            if not 0 <= frame < job_size:
                raise XmlError(
                    f"frame {frame} outside the job range 0..{job_size - 1} "
                    f"(track '{label}')"
                )

            left, top = int(m.get("left")), int(m.get("top"))
            width, height = int(m.get("width")), int(m.get("height"))
            rle = m.get("rle")
            if rle is None:
                raise XmlError(f"frame {frame}, track '{label}': outside=0 but no rle")

            counts = [int(x) for x in rle.split(",")]
            if sum(counts) != width * height:
                raise XmlError(
                    f"frame {frame}, track '{label}': sum(rle)={sum(counts)} != "
                    f"width*height={width * height}"
                )
            if left < 0 or top < 0 or left + width > frame_width or top + height > frame_height:
                raise XmlError(
                    f"frame {frame}, track '{label}': bbox "
                    f"[{left},{top},{left + width},{top + height}] "
                    f"outside the frame {frame_width}x{frame_height}"
                )

            geometry.append((rle, width, height))
            entries.append({
                "frame": frame,
                "label": label,
                "group": group_idx,
                "counts": counts,
                "left": left, "top": top, "width": width, "height": height,
                "occluded": m.get("occluded") == "1",
                "z_order": int(m.get("z_order") or 0),
                "area": rle_area(counts),
            })

    if not entries:
        raise XmlError("every mask is outside=1: nothing to upload")

    if not skip_rle_check:
        st = orientation_stats(geometry)
        if st["as_col_major"] < st["as_is"]:
            raise XmlError(
                f"RLE is column-major: {st['as_is']:.2f} runs per row as written, "
                f"{st['as_col_major']:.2f} when read by columns, so the masks are more "
                f"connected that way. CVAT reads row by row, the UI would show "
                f"diagonal stripes. Re-export the XML, "
                f"or pass --skip-rle-check"
            )

    return entries


# ── verification ────────────────────────────────────────────────────────────────────

def key_of_entry(e: dict) -> tuple:
    return (e["frame"], e["label"], e["group"], e["area"])


def verify(job, entries: list[dict], label_names: dict[int, str],
           kept_manual: int) -> list[str]:
    """Read the annotation back and compare it with what was sent."""
    ann = job.get_annotations()
    problems: list[str] = []

    server = []
    manual = 0
    for s in ann.shapes:
        if getattr(s, "source", None) == "manual" and s.group == 0:
            manual += 1
            continue
        counts = [int(round(v)) for v in s.points[:-4]]
        server.append((s.frame, label_names.get(s.label_id, str(s.label_id)),
                       s.group, rle_area(counts)))

    if ann.tracks:
        problems.append(f"tracks left on the server: {len(ann.tracks)}")

    want_frames = {e["frame"] for e in entries}
    got_frames = {t[0] for t in server}
    if want_frames != got_frames:
        missing = sorted(want_frames - got_frames)
        extra = sorted(got_frames - want_frames)
        detail = []
        if missing:
            detail.append(f"frames missing ({len(missing)}): {missing[:10]}"
                          f"{' …' if len(missing) > 10 else ''}")
        if extra:
            detail.append(f"extra frames ({len(extra)}): {extra[:10]}"
                          f"{' …' if len(extra) > 10 else ''}")
        problems.append("; ".join(detail))

    from collections import Counter
    want = Counter(key_of_entry(e) for e in entries)
    got = Counter(server)
    if want != got:
        lost = sum((want - got).values())
        gained = sum((got - want).values())
        problems.append(
            f"shape mismatch: {lost} missing on the server, {gained} extra "
            f"(compared: frame + label + group + mask area)"
        )

    if manual != kept_manual:
        problems.append(f"{manual} manual shapes on the server, expected {kept_manual}")

    return problems


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xml-dir", type=Path, required=True, help="folder with job_*.xml")
    ap.add_argument("--jobs", nargs="*", type=int, help="job ids")
    ap.add_argument("--all", action="store_true", help="every XML in --xml-dir")
    ap.add_argument("--backup-dir", type=Path, default=Path("preanno_backup"),
                    help="where the JSON of the current annotation goes before it is replaced")
    ap.add_argument("--dry-run", action="store_true",
                    help="only checks and report, write nothing")
    ap.add_argument("--drop-manual", action="store_true",
                    help="discard shapes with source=manual too (kept by default)")
    ap.add_argument("--no-verify", action="store_true", help="do not read back and compare after the upload")
    ap.add_argument("--skip-rle-check", action="store_true",
                    help="skip the RLE orientation check")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.xml_dir.glob("job_*.xml"), key=lambda p: int(p.stem.split("_")[1]))
    if not args.all:
        if not args.jobs:
            ap.error("give --all or --jobs")
        wanted = set(args.jobs)
        files = [p for p in files if int(p.stem.split("_")[1]) in wanted]
    if not files:
        print("no XML found", file=sys.stderr)
        return 1

    print(f"to upload: {len(files)} job(s) from {args.xml_dir}")
    if args.dry_run:
        print("DRY RUN: nothing will be written\n")

    login = os.environ.get("CVAT_LOGIN") or input("CVAT login: ").strip()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass("CVAT password: ")

    if args.host:
        host, port = args.host, args.port
    else:
        host, port = resolve_cvat_host()

    ok, failed, skipped = [], [], []

    with make_client(host=host, port=port, credentials=(login, password)) as client:
        client.organization_slug = None
        print("connected\n")

        for xml_path in files:
            job_id = int(xml_path.stem.split("_")[1])
            try:
                job = client.jobs.retrieve(job_id)
                meta = job.get_meta()
                frame = meta.frames[0]
                job_size = job.stop_frame - job.start_frame + 1

                xml_size = ET.parse(xml_path).getroot().findtext("meta/job/size")
                if xml_size and int(xml_size) != job_size:
                    raise XmlError(
                        f"size mismatch: {xml_size} frames in the XML, {job_size} in the job. "
                        f"The XML probably belongs to another job"
                    )

                entries = parse_job_xml(
                    xml_path, frame.width, frame.height, job_size, args.skip_rle_check
                )

                # labels: by name only, nothing is created
                task = client.tasks.retrieve(job.task_id)
                owner = (client.projects.retrieve(task.project_id)
                         if task.project_id else task)
                labels = {l.name: l.id for l in owner.get_labels()}
                names = {v: k for k, v in labels.items()}
                need = {e["label"] for e in entries}
                absent = sorted(need - labels.keys())
                if absent:
                    raise XmlError(
                        f"labels missing in the project: {absent}. Create them in CVAT and retry"
                    )

                # what the job holds now
                existing = job.get_annotations()
                manual_shapes = [
                    s for s in existing.shapes
                    if getattr(s, "source", None) == "manual" and s.group == 0
                ] if not args.drop_manual else []

                frames = sorted({e["frame"] for e in entries})
                per_label = {}
                for e in entries:
                    per_label[e["label"]] = per_label.get(e["label"], 0) + 1

                print(f"job_{job_id}: {len(entries)} masks on {len(frames)} frames "
                      f"(0..{job_size - 1}), {len(need)} instance(s)")
                print(f"    labels: {per_label}")
                if len(frames) != job_size:
                    gaps = sorted(set(range(job_size)) - set(frames))
                    print(f"    frames without masks: {len(gaps)} "
                          f"{gaps[:10]}{' …' if len(gaps) > 10 else ''}")
                if existing.shapes or existing.tracks:
                    print(f"    will replace: {len(existing.shapes)} shapes, "
                          f"{len(existing.tracks)} tracks"
                          + (f"; keeping {len(manual_shapes)} manual"
                             if manual_shapes else ""))

                if args.dry_run:
                    ok.append(job_id)
                    continue

                # backup of the current state
                if existing.shapes or existing.tracks:
                    args.backup_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bak = args.backup_dir / f"job_{job_id}_{ts}.json"
                    bak.write_text(
                        json.dumps(existing.to_dict(), ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
                    print(f"    backup -> {bak}")

                shapes = [
                    LabeledShapeRequest(
                        type=ShapeType("mask"),
                        frame=e["frame"],
                        label_id=labels[e["label"]],
                        group=e["group"],
                        points=[float(v) for v in e["counts"] + [
                            e["left"], e["top"],
                            e["left"] + e["width"] - 1, e["top"] + e["height"] - 1,
                        ]],
                        occluded=e["occluded"],
                        z_order=e["z_order"],
                    )
                    for e in entries
                ]
                shapes += [
                    LabeledShapeRequest(
                        type=s.type, frame=s.frame, label_id=s.label_id, group=s.group,
                        points=s.points, occluded=s.occluded, z_order=s.z_order,
                        source="manual",
                    )
                    for s in manual_shapes
                ]

                job.set_annotations(LabeledDataRequest(shapes=shapes, tracks=[], tags=[]))

                if args.no_verify:
                    print(f"    uploaded {len(shapes)} shapes (no verification)")
                    ok.append(job_id)
                    continue

                problems = verify(job, entries, names, len(manual_shapes))
                if problems:
                    for p in problems:
                        print(f"    VERIFY FAILED: {p}", file=sys.stderr)
                    failed.append(job_id)
                else:
                    print(f"    verified: {len(entries)} masks on "
                          f"{len(frames)} frames, areas match"
                          + (f", {len(manual_shapes)} manual kept"
                             if manual_shapes else ""))
                    ok.append(job_id)

            except XmlError as exc:
                print(f"job_{job_id}: FAILED {exc}", file=sys.stderr)
                failed.append(job_id)
            except Exception as exc:
                print(f"job_{job_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
                failed.append(job_id)
            print()

    verb = "checked" if args.dry_run else "uploaded"
    print(f"done: {verb} {len(ok)}, failed {len(failed)} of {len(files)}")
    if failed:
        print(f"   failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
