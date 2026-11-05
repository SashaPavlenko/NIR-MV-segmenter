#!/bin/bash
# Reproduce the main result of the thesis: the final method (NIR + disparity ensemble,
# overlapping windows, cross-view consensus) evaluated leave-one-participant-out.
#
# Every fold model predicts the recordings of the participant it was not trained on
# (splits/folds.json: infer_jobs = all eight views of those recordings, test_jobs = the
# annotated ones). The 52 annotated views of the five folds are then scored together.
#
#     bash scripts/reproduce_thesis.sh                     # one GPU process at a time
#     PARALLEL=2 bash scripts/reproduce_thesis.sh          # two at a time (about 31 GB GPU memory each)
#
# Expected (thesis, final method): mIoU over the seven classes = 0.656 on 52 views.
# Output: outputs/reproduce/eval/summary.md
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_env
OUT="${OUT:-$OUTPUT_ROOT/reproduce}"
PARALLEL="${PARALLEL:-1}"
WORKERS="${WORKERS:-8}"
FOLDS="${FOLDS:-1 2 3 4 5}"
mkdir -p "$OUT"; OUT="$(realpath "$OUT")"

ALL_INFER=""; ALL_TEST=""
for k in $FOLDS; do
    weights_of combo_nir "$k" > /dev/null; weights_of combo_depth "$k" > /dev/null
    ALL_INFER="$ALL_INFER $(fold_field "$k" infer_jobs)"
    ALL_TEST="$ALL_TEST $(fold_field "$k" test_jobs)"
done
log "folds: $FOLDS | $(wc -w <<< "$ALL_INFER") views to predict, $(wc -w <<< "$ALL_TEST") annotated"

log "1/5 disparity frames"
python "$SRC/precompute_depth.py" --job-frames "$JOB_FRAMES" --recordings "$RECORDINGS_DIR" \
    --out "$OUT/depth_frames" --jobs $ALL_INFER --workers "$WORKERS"

log "2/5 fold models ($PARALLEL at a time)"
mkdir -p "$OUT/logs"
cd "$HOIST_DIR"
run_one() {   # <model> <fold>
    local m="$1" k="$2" root ext
    if [ "$m" = combo_nir ]; then root="$RECORDINGS_DIR"; ext=.bmp; else root="$OUT/depth_frames"; ext=.png; fi
    python "$SRC/infer_soft.py" --config-file "$CONFIG" --weights "$(weights_of "$m" "$k")" \
        --job-frames "$JOB_FRAMES" --data-root "$root" --ext "$ext" --jobs $(fold_field "$k" infer_jobs) \
        --out "$OUT/soft_${m#combo_}" > "$OUT/logs/${m}_fold$k.log" 2>&1 \
        && echo "    $m fold $k done" || { echo "    $m fold $k FAILED, see $OUT/logs/${m}_fold$k.log"; return 1; }
}
pids=()
for k in $FOLDS; do
    for m in combo_nir combo_depth; do
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
        run_one "$m" "$k" & pids+=($!)
        sleep 2
    done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "some inference runs failed" >&2; exit 1; }
cd "$REPO"

log "3/5 ensemble"
python "$SRC/ensemble.py" --soft "$OUT/soft_nir" "$OUT/soft_depth" --out "$OUT/ensemble" --workers "$WORKERS" > "$OUT/logs/ensemble.log"
tail -1 "$OUT/logs/ensemble.log"

log "4/5 cross-view consensus"
python "$SRC/build_view_index.py" --job-frames "$JOB_FRAMES" --out "$OUT/view_index.json" > "$OUT/logs/view_index.log"
python "$SRC/crossview_consensus.py" --index "$OUT/view_index.json" --raw-in "$OUT/ensemble" \
    --raw-out "$OUT/consensus" --report "$OUT/consensus_report.json" > "$OUT/logs/consensus.log"

log "5/5 scores"
python "$SRC/evaluate.py" --pred "$OUT/consensus" --jobs $ALL_TEST --out "$OUT/eval" --workers "$WORKERS"
echo
echo "thesis (final method, 52 views): mIoU 7 classes = 0.656"
log "done: $OUT/eval/summary.md"
