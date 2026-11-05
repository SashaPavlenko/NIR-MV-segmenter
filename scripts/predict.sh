#!/bin/bash
# Final method of the thesis on whole recordings (all eight views of each recording):
#
#   1. disparity frames from the two eyes of every device           (CPU)
#   2. NIR model and NIR + disparity model, 16-frame windows, stride 8 (GPU)
#   3. ensemble of the two models, binary masks, small-component filter
#   4. cross-view consensus over the eight views of a recording       (CPU)
#   5. CVAT for images 1.1 XML per view (for cvat/upload_preannotations.py)
#   6. scores, if some of the views are annotated in data/annotations
#
#     bash scripts/predict.sh --runs "run_1_drill_3 run_1_scalpel_2" --fold 1 --out outputs/predict
#
# Options (all optional):
#   --views FILE        view list, default data/job_frames.json (or from src/make_views.py)
#   --recordings DIR    default data/recordings
#   --runs "A B"        recordings to process, default: every recording in the view list
#   --fold K            which of the five fold models to use (1-5), default 1
#   --out DIR           default outputs/predict
#   --workers N         CPU processes for steps 1, 3, 6 (default 8)
# Steps whose outputs exist are skipped; delete the output folder to start over.
set -euo pipefail
source "$(dirname "$0")/common.sh"

VIEWS="$JOB_FRAMES"; RECS="$RECORDINGS_DIR"; RUNS=""; FOLD=1; OUT="$OUTPUT_ROOT/predict"; WORKERS=8
while [ $# -gt 0 ]; do
    case "$1" in
        --views) VIEWS="$2"; shift 2 ;;
        --recordings) RECS="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) sed -n 2,21p "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done
activate_env
VIEWS="$(realpath "$VIEWS")"; RECS="$(realpath "$RECS")"; mkdir -p "$OUT"; OUT="$(realpath "$OUT")"
W_NIR="$(weights_of combo_nir "$FOLD")"
W_DEPTH="$(weights_of combo_depth "$FOLD")"

# view ids of the selected recordings
JOBS=$(python - "$VIEWS" "$RUNS" <<'EOF'
import json, sys
views = json.load(open(sys.argv[1]))
runs = set(sys.argv[2].split())
ids = [v["job_id"] for v in views if not runs or v["task_name"].split("__")[0] in runs]
missing = runs - {v["task_name"].split("__")[0] for v in views}
if missing:
    sys.exit(f"not in the view list: {sorted(missing)}")
print(" ".join(map(str, ids)))
EOF
)
log "$(wc -w <<< "$JOBS") views, fold-$FOLD models, output $OUT"

log "1/6 disparity frames"
python "$SRC/precompute_depth.py" --job-frames "$VIEWS" --recordings "$RECS" \
    --out "$OUT/depth_frames" --jobs $JOBS --workers "$WORKERS"

cd "$HOIST_DIR"
log "2/6 NIR model ($W_NIR)"
python "$SRC/infer_soft.py" --config-file "$CONFIG" --weights "$W_NIR" --job-frames "$VIEWS" \
    --data-root "$RECS" --ext .bmp --jobs $JOBS --out "$OUT/soft_nir"
log "2/6 NIR + disparity model ($W_DEPTH)"
python "$SRC/infer_soft.py" --config-file "$CONFIG" --weights "$W_DEPTH" --job-frames "$VIEWS" \
    --data-root "$OUT/depth_frames" --ext .png --jobs $JOBS --out "$OUT/soft_depth"
cd "$REPO"

log "3/6 ensemble"
python "$SRC/ensemble.py" --soft "$OUT/soft_nir" "$OUT/soft_depth" --out "$OUT/ensemble" --workers "$WORKERS" > "$OUT/ensemble.log"
tail -1 "$OUT/ensemble.log"

log "4/6 cross-view consensus"
python "$SRC/build_view_index.py" --job-frames "$VIEWS" --out "$OUT/view_index.json" > "$OUT/view_index.log"
python "$SRC/crossview_consensus.py" --index "$OUT/view_index.json" --raw-in "$OUT/ensemble" \
    --raw-out "$OUT/consensus" --report "$OUT/consensus_report.json" ${RUNS:+--runs $RUNS} > "$OUT/consensus.log"

log "5/6 CVAT XML"
python "$SRC/postprocess_and_export.py" --raw-out "$OUT/consensus" --out-dir "$OUT/cvat_xml" --jobs $JOBS > "$OUT/export.log"
echo "    $(ls "$OUT/cvat_xml" | wc -l) files in $OUT/cvat_xml"

log "6/6 scores on annotated views"
ANNOTATED=$(for j in $JOBS; do [ -f "$ANNOTATIONS_DIR/job_$j.xml" ] && echo -n "$j "; done; true)
if [ -n "$ANNOTATED" ]; then
    python "$SRC/evaluate.py" --pred "$OUT/consensus" --job-frames "$VIEWS" --jobs $ANNOTATED \
        --out "$OUT/eval" --workers "$WORKERS"
else
    echo "    no annotated view among them"
fi
log "done: $OUT"
