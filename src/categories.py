"""Single source of truth for the 8 CVAT category names and their order.

Order matters: it fixes the category_id (1-based, for YTVIS JSON) <->
contiguous id (0-based, the model's classifier output index) mapping used
consistently across cvat_to_ytvis.py, register_finetune_dataset.py, and the
inference/export scripts.
"""

CATEGORIES = [
    "hand_left",
    "hand_right",
    "haptic_left",
    "haptic_right",
    "scalpel",
    "screwdriver",
    "drill",
    "object",
]

# Classes that are physically continuous held tools/devices -> eligible for
# the "largest connected component" post-processing step at inference time.
TOOL_CATEGORIES = {
    "haptic_left",
    "haptic_right",
    "scalpel",
    "screwdriver",
    "drill",
}

CAT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CATEGORIES)}
CAT_ID_TO_NAME = {i + 1: name for i, name in enumerate(CATEGORIES)}
