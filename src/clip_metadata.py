"""clip_metadata.py — per-clip (per-job) conditioning metadata: activity type
and camera position, derived from task_name (run_<n>_<activity>_<inst>__cam<N>_<side>).

Single source of truth for the id<->name mapping used on both sides:
  - cvat_to_ytvis.py writes activity_id/camera_pos_id into
    each YTVIS "video" entry;
  - HOISTFormer/mask2former_video/config.py's NUM_ACTIVITY_CLASSES /
    NUM_CAMERA_POSITIONS must match len(ACTIVITIES) / len(CAMERA_POSITIONS)
    below (the model only needs the counts + the raw ints from the JSON, but
    keep them in sync if you add a category here).

Camera positions: cam1 and cam2 sit on
the table facing the ceiling; cam3 faces the person at an angle, mounted
slightly higher, angled down at the hands; cam4 is mounted highest of all,
facing down at the floor. cam1/cam2 are grouped into one "table_up" position
since they share the same physical placement/orientation; left/right in the
task_name is the stereo side of a camera, not a different physical position.
"""

import re

TASK_NAME_RE = re.compile(r"^run_(\d+)_(\w+)_(\d+)__cam(\d+)_(left|right)$")

ACTIVITIES = ["idle", "drill", "scalpel", "screwdriver"]
ACTIVITY_TO_ID = {name: i for i, name in enumerate(ACTIVITIES)}

CAMERA_POSITIONS = ["table_up", "front_angled", "top_down"]
CAM_NUMBER_TO_POSITION_ID = {1: 0, 2: 0, 3: 1, 4: 2}  # cam# -> index into CAMERA_POSITIONS


class TaskNameError(Exception):
    pass


def parse_task_name(task_name: str) -> dict:
    m = TASK_NAME_RE.match(task_name)
    if not m:
        raise TaskNameError(f"unexpected task_name format: {task_name!r}")
    run, activity, instance, cam, side = m.groups()
    return {
        "run": int(run),
        "activity": activity,
        "instance": int(instance),
        "cam": int(cam),
        "side": side,
    }


def activity_id_of(task_name: str) -> int:
    """-1 if the activity isn't one of the known ACTIVITIES (fails loud instead:
    raises, since a genuinely new activity name should be added above, not
    silently bucketed as unknown)."""
    activity = parse_task_name(task_name)["activity"]
    if activity not in ACTIVITY_TO_ID:
        raise TaskNameError(
            f"task_name {task_name!r}: activity {activity!r} not in {ACTIVITIES} "
            f"-- add it to clip_metadata.ACTIVITIES if this is a real new category"
        )
    return ACTIVITY_TO_ID[activity]


def camera_pos_id_of(task_name: str) -> int:
    cam = parse_task_name(task_name)["cam"]
    if cam not in CAM_NUMBER_TO_POSITION_ID:
        raise TaskNameError(
            f"task_name {task_name!r}: cam{cam} has no known physical position "
            f"-- add it to clip_metadata.CAM_NUMBER_TO_POSITION_ID"
        )
    return CAM_NUMBER_TO_POSITION_ID[cam]
