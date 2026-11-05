from __future__ import annotations

import argparse
import getpass
import json
import os
import re
from itertools import zip_longest
from pathlib import Path

from cvat_host import resolve_cvat_host
from cvat_sdk import make_client


def parse_frame_step(frame_filter: str | None) -> int:
    if not frame_filter:
        return 1

    match = re.search(r"step\s*=\s*([1-9]\d*)", frame_filter)
    if match is None:
        return 1

    return int(match.group(1))


def get_job_frame_ids(meta) -> list[int]:

    if meta.included_frames:
        return list(meta.included_frames)

    step = parse_frame_step(getattr(meta, "frame_filter", None))
    start_frame = getattr(meta, "start_frame", 0)
    stop_frame = getattr(meta, "stop_frame", start_frame)

    return list(range(start_frame, stop_frame + 1, step))


def build_record(job) -> dict:
    meta = job.get_meta()
    frame_ids = get_job_frame_ids(meta)
    frame_names = [frame.name for frame in (meta.frames or [])]
    frame_entries = [
        {
            "frame_id": frame_id,
            "file_name": frame_name,
        }
        for frame_id, frame_name in zip_longest(frame_ids, frame_names)
    ]

    return {
        "job_id": job.id,
        "job_name": getattr(job, "name", None),
        "task_id": job.task_id,
        "task_name": getattr(job, "task_name", None),
        "job_type": getattr(job, "type", None),
        "frame_count": len(frame_entries),
        "frame_ids": frame_ids,
        "frame_names": frame_names,
        "frames": frame_entries,
        "frame_filter": getattr(meta, "frame_filter", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the frame list of every CVAT job into job_frames.json "
                    "(the view list of the pipeline: job id, task name, frame ids and file names)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("job_frames.json"),
        help="Path to the JSON file that will be created.",
    )
    args = parser.parse_args()

    print("=== CVAT Job Frame Exporter ===")
    login = os.environ.get("CVAT_LOGIN") or input("CVAT login: ").strip()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass("CVAT password: ")

    cvat_host, cvat_port = resolve_cvat_host()
    print("connecting...")

    with make_client(host=cvat_host, port=cvat_port, credentials=(login, password)) as client:
        print("connected\n")

        jobs = sorted(client.jobs.list(), key=lambda job: job.id)
        print(f"{len(jobs)} jobs")

        records = []
        for job in jobs:
            record = build_record(job)
            records.append(record)
            print(
                f"#{record['job_id']} {record['job_name']} -> "
                f"{record['frame_count']} frames"
            )

    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())