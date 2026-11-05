#!/bin/bash
# Check that data/ and weights/ are complete before running anything heavy.
#     bash scripts/check_data.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
activate_env > /dev/null 2>&1 || true
PY="$(command -v python || command -v python3)"   # only the standard library is needed here
ok=1
"$PY" - "$JOB_FRAMES" "$RECORDINGS_DIR" "$ANNOTATIONS_DIR" "$FOLDS_JSON" <<'EOF' || ok=0
import json, sys
from pathlib import Path
jf, rec, ann, folds = map(Path, sys.argv[1:])
bad = False
if not jf.exists():
    sys.exit(f"MISSING view list {jf}")
views = json.loads(jf.read_text())
missing_frames, missing_views = 0, []
for v in views:
    run, cam = v["task_name"].split("__")
    d = rec / run / "images" / cam[3] / cam.split("_")[1]
    if not d.is_dir():
        missing_views.append(v["task_name"])
        continue
    missing_frames += sum(not (d / n).exists() for n in v["frame_names"])
print(f"view list: {len(views)} views in {len({v['task_name'].split('__')[0] for v in views})} recordings")
if missing_views:
    bad = True
    print(f"MISSING frame folders for {len(missing_views)} views, e.g. {missing_views[:3]}")
if missing_frames:
    bad = True
    print(f"MISSING {missing_frames} frames listed in {jf.name}")
xml = sorted(ann.glob("job_*.xml"))
print(f"annotations: {len(xml)} views in {ann}")
need = {j for f in json.loads(folds.read_text()).values() for j in f["test_jobs"]}
have = {int(p.stem.split("_")[1]) for p in xml}
if need - have:
    bad = True
    print(f"MISSING annotations of {len(need - have)} test views of splits/folds.json")
ids = {v["job_id"] for v in views}
need_views = {j for f in json.loads(folds.read_text()).values() for j in f["infer_jobs"]}
if need_views - ids:
    bad = True
    print(f"MISSING {len(need_views - ids)} views of splits/folds.json in {jf.name}")
sys.exit(1 if bad else 0)
EOF
for m in combo_nir combo_depth; do
    for k in 1 2 3 4 5; do
        w="$WEIGHTS_DIR/$m/fold$k.pth"
        if [ -f "$w" ]; then
            echo "weights: $w ($(du -Lh "$w" | cut -f1))"
        else
            echo "MISSING weights $w"; ok=0
        fi
    done
done
[ -f "$HOIST_DIR/configs/hoist/v4_combo.yaml" ] || echo "note: HOIST-Former is not installed yet (bash scripts/setup.sh)"
if [ "$ok" = 1 ]; then echo "all complete"; else echo "incomplete, see docs/2_dataset.md and docs/3_weights.md"; exit 1; fi
