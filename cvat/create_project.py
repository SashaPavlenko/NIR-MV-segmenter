"""Create the CVAT project with the eight mask labels of the thesis.

    python create_project.py --name "NIR hands and tools"

Prints the project id (needed by create_tasks.py). The label names must stay exactly as they
are: the pipeline and upload_preannotations.py find the labels by name.
Login: CVAT_LOGIN / CVAT_PASSWORD from the environment, otherwise asked for.
"""
import argparse
import getpass
import os

from cvat_sdk import make_client, models

from cvat_host import resolve_cvat_host

LABELS = [  # order = model class index (src/categories.py)
    ("hand_left", "#33ddff"),
    ("hand_right", "#fa3253"),
    ("haptic_left", "#34d1b7"),
    ("haptic_right", "#ff007c"),
    ("scalpel", "#ddff33"),
    ("screwdriver", "#24b353"),
    ("drill", "#b83df5"),
    ("object", "#66ff66"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="NIR hands and tools")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    login = os.environ.get("CVAT_LOGIN") or input("CVAT login: ").strip()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass("CVAT password: ")
    host, port = (a.host, a.port) if a.host else resolve_cvat_host()
    with make_client(host=host, port=port, credentials=(login, password)) as client:
        project = client.projects.create(models.ProjectWriteRequest(
            name=a.name,
            labels=[models.PatchedLabelRequest(name=n, color=c, type="mask") for n, c in LABELS]))
        print(f"project #{project.id} '{project.name}' with labels {[n for n, _ in LABELS]}")


if __name__ == "__main__":
    main()
