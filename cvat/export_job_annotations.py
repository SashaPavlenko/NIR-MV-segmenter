"""export_job_annotations.py -- download the annotation of CVAT jobs as "CVAT for images 1.1" XML.

One file per job, job_<id>.xml, in a flat folder: this is the layout of data/annotations/
(masks stay masks, RLE inside the XML). Annotation only, no frames: the originals are in
data/recordings (--with-images to get them anyway; the job is then unpacked into job_<id>/).

    export CVAT_LOGIN=<user> CVAT_PASSWORD=<password>
    python3 export_job_annotations.py 390 391 409 --output ../data/annotations
    python3 export_job_annotations.py --jobs-file jobs.txt --output ../data/annotations

Jobs whose XML is already in --output are skipped (--force to download again).
Talk to CVAT directly (localhost), not through an ngrok tunnel.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import zipfile
from pathlib import Path

from cvat_sdk import make_client

from cvat_host import resolve_cvat_host

FORMAT = "CVAT for images 1.1"


def read_jobs_file(path: Path) -> list[int]:
    """Job ids from a file: one or more per line, empty lines and # comments ignored."""
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            jobs.extend(int(tok) for tok in line.replace(",", " ").split())
    return jobs


def extract(zip_path: Path, out_dir: Path, job_id: int, with_images: bool) -> None:
    """Annotation -> job_<id>.xml; with images: the whole zip -> job_<id>/."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if with_images:
            unpacked = out_dir / f"job_{job_id}"
            zf.extractall(unpacked)
            print(f"    unpacked into {unpacked.name}/ ({len(names)} files)")
            return
        xml_names = [n for n in names if n.endswith("annotations.xml")]
        if not xml_names:
            raise RuntimeError(f"no annotations.xml in the zip: {names[:5]}")
        xml_path = out_dir / f"job_{job_id}.xml"
        xml_path.write_bytes(zf.read(xml_names[0]))
        print(f"    saved {xml_path.name} ({xml_path.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", type=int, help="job ids")
    ap.add_argument("--jobs-file", type=Path, help="file with job ids")
    ap.add_argument("--output", type=Path, required=True, help="folder for the XML files")
    ap.add_argument("--host", default=None, help="default: cvat_host.py (localhost:8181)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--format", default=FORMAT, help=f"export format (default: {FORMAT})")
    ap.add_argument("--with-images", action="store_true", help="also download the frames")
    ap.add_argument("--force", action="store_true", help="download again even if the XML exists")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip files")
    args = ap.parse_args()

    jobs = list(args.jobs)
    if args.jobs_file:
        jobs.extend(read_jobs_file(args.jobs_file))
    if not jobs:
        ap.error("no job given (ids or --jobs-file)")
    jobs = sorted(dict.fromkeys(jobs))

    login = os.environ.get("CVAT_LOGIN") or input("CVAT login: ").strip()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass("CVAT password: ")
    host, port = (args.host, args.port) if args.host else resolve_cvat_host()

    args.output.mkdir(parents=True, exist_ok=True)
    zip_dir = args.output / "_zip"
    zip_dir.mkdir(exist_ok=True)
    done, skipped, failed = 0, 0, []

    with make_client(host=host, port=port, credentials=(login, password)) as client:
        client.organization_slug = None
        print(f"connected, {len(jobs)} job(s) to export\n")
        for job_id in jobs:
            target = (args.output / f"job_{job_id}") if args.with_images else (args.output / f"job_{job_id}.xml")
            if target.exists() and not args.force:
                print(f"[skip] job_{job_id}: {target.name} exists")
                skipped += 1
                continue
            zip_path = zip_dir / f"job_{job_id}.zip"
            try:
                job = client.jobs.retrieve(job_id)
                print(f"[export] job_{job_id} (task {job.task_id}, frames {job.start_frame}-{job.stop_frame})")
                job.export_dataset(args.format, str(zip_path), include_images=args.with_images,
                                   status_check_period=2)
                extract(zip_path, args.output, job_id, args.with_images)
                done += 1
            except Exception as exc:
                print(f"  FAILED job_{job_id}: {exc}", file=sys.stderr)
                failed.append(job_id)
            finally:
                if zip_path.exists() and not args.keep_zip:
                    zip_path.unlink()

    if not args.keep_zip and not any(zip_dir.iterdir()):
        zip_dir.rmdir()
    print(f"\ndone: {done} exported, {skipped} skipped, {len(failed)} failed -> {args.output}")
    if failed:
        print(f"failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
