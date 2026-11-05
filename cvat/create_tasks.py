"""Create one CVAT task per view (recording x device x eye) of the recordings to annotate.

    python create_tasks.py --project-id 1 --recordings ../data/recordings --runs-csv annotation_runs.csv

--runs-csv lists the recordings (columns name, annotate, fps; see annotation_runs.example.csv).
Only rows with annotate=yes are used. The recordings were captured at --source-fps (90); CVAT
keeps every step-th frame with step = round(source_fps / fps), so fps=30 gives step 3.
Task name: <run>__cam<device>_<side>, the name the whole pipeline relies on.
The labels come from the project (create it first, see docs/7_cvat.md).
Login: CVAT_LOGIN / CVAT_PASSWORD from the environment, otherwise asked for.
"""
import argparse
import csv
import getpass
import os
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.models import TaskWriteRequest
from tqdm import tqdm

from cvat_host import resolve_cvat_host

CAMERAS = ["1", "2", "3", "4"]
SIDES = ["left", "right"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", type=int, required=True, help="see the project URL: /projects/<id>")
    ap.add_argument("--recordings", type=Path, required=True, help="folder with one sub-folder per recording")
    ap.add_argument("--runs-csv", type=Path, required=True)
    ap.add_argument("--source-fps", type=int, default=90)
    ap.add_argument("--image-quality", type=int, default=80)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()

    with open(a.runs_csv, newline="") as f:
        runs = {row["name"]: int(row["fps"]) for row in csv.DictReader(f)
                if row["annotate"].strip().lower() == "yes"}
    print(f"{len(runs)} recordings with annotate=yes:")
    for name, fps in sorted(runs.items()):
        print(f"  {name}  ({fps} fps -> frame step {round(a.source_fps / fps)})")

    login = os.environ.get("CVAT_LOGIN") or input("CVAT login: ").strip()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass("CVAT password: ")
    host, port = (a.host, a.port) if a.host else resolve_cvat_host()

    with make_client(host=host, port=port, credentials=(login, password)) as client:
        run_dirs = sorted(d for d in a.recordings.iterdir() if d.is_dir() and d.name in runs)
        missing = sorted(set(runs) - {d.name for d in run_dirs})
        if missing:
            print(f"not found in {a.recordings}: {missing}")
        combos = [(d, cam, side) for d in run_dirs for cam in CAMERAS for side in SIDES]
        errors = []
        for run_dir, cam, side in tqdm(combos, unit="task", ncols=90):
            name = f"{run_dir.name}__cam{cam}_{side}"
            img_dir = run_dir / "images" / cam / side
            images = sorted(f for f in img_dir.glob("*") if f.suffix in (".bmp", ".png", ".jpg")) if img_dir.is_dir() else []
            if not images:
                tqdm.write(f"no images in {img_dir}")
                errors.append(name)
                continue
            step = round(a.source_fps / runs[run_dir.name])
            try:
                task = client.tasks.create(TaskWriteRequest(name=name, project_id=a.project_id, subset="train"))
                task.upload_data(resources=[str(f) for f in images],
                                 params={"image_quality": a.image_quality, "frame_step": step,
                                         "sorting_method": "natural"})
                kept = (len(images) + step - 1) // step
                tqdm.write(f"task #{task.id} {name}: {len(images)} -> {kept} frames (step {step})")
            except Exception as e:  # keep going, report at the end
                tqdm.write(f"FAILED {name}: {e}")
                errors.append(name)
    print(f"\ndone: {len(combos) - len(errors)}/{len(combos)} tasks" + (f", failed: {errors}" if errors else ""))


if __name__ == "__main__":
    main()
