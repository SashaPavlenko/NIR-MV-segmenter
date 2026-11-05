#!/bin/bash
# Train the final models (config v4_combo: 768 px, 3x loss weight on the tools, brightness and
# contrast augmentation), leave-one-participant-out as in the thesis, or on all annotated views.
#
#     MODEL=combo_nir   bash scripts/train_lopo.sh            # five folds, NIR input
#     MODEL=combo_depth bash scripts/train_lopo.sh            # five folds, NIR + disparity input
#     MODEL=combo_nir FOLDS=all bash scripts/train_lopo.sh    # one model on all 52 annotated views
#     PARALLEL=3 ...                                          # three folds at once (about 23 GB GPU memory each)
#     EXTRA_OPTS="SOLVER.MAX_ITER 20" ...                     # short smoke test
#
# Needs the original HOIST-Former weights (scripts/setup.sh --pretrained).
# Output: outputs/train/<MODEL>/fold<k>/ ; the checkpoint to use is named in its last_checkpoint
# file. Copy it to weights/<MODEL>/fold<k>.pth to use it with predict.sh / reproduce_thesis.sh.
# Training is not bit-exact: a new run gives slightly different weights than the thesis models.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_env
MODEL="${MODEL:?set MODEL=combo_nir or MODEL=combo_depth}"
FOLDS="${FOLDS:-1 2 3 4 5}"
PARALLEL="${PARALLEL:-1}"
OUT="${OUT:-$OUTPUT_ROOT/train/$MODEL}"
mkdir -p "$OUT"; OUT="$(realpath "$OUT")"
[ -f "$HOIST_DIR/pretrained_models/trained_model.pth" ] || { echo "missing HOIST-Former weights: bash scripts/setup.sh --pretrained" >&2; exit 1; }

ALL=$(ls "$ANNOTATIONS_DIR" | sed -n 's/^job_\([0-9]*\)\.xml$/\1/p' | sort)
case "$MODEL" in
    combo_nir) ROOT="$RECORDINGS_DIR"; EXT=.bmp ;;
    combo_depth)
        ROOT="$OUTPUT_ROOT/train/depth_frames"; EXT=.png
        log "disparity frames of the annotated views"
        python "$SRC/precompute_depth.py" --job-frames "$JOB_FRAMES" --recordings "$RECORDINGS_DIR" \
            --out "$ROOT" --jobs $ALL ;;
    *) echo "MODEL must be combo_nir or combo_depth" >&2; exit 1 ;;
esac

train_fold() {   # <k | all>
    local k="$1" hold train d o
    if [ "$k" = all ]; then hold=""; else hold=$(fold_field "$k" test_jobs); fi
    train=$(comm -23 <(echo "$ALL") <(tr ' ' '\n' <<< "$hold" | sort) | tr '\n' ' ')
    d="$OUT/fold$k"; o="$d"; mkdir -p "$d"
    python "$SRC/cvat_to_ytvis.py" --job-frames "$JOB_FRAMES" --cvat-xml-dir "$ANNOTATIONS_DIR" \
        --data-root "$RECORDINGS_DIR" --out "$d/train.json" --jobs $train > "$d/ytvis.log" 2>&1
    # the depth model reads the disparity frames instead of the recordings
    if [ "$EXT" = .png ]; then
        python - "$d/train.json" "$RECORDINGS_DIR" "$ROOT" <<'EOF'
import json, sys
p, src, dst = sys.argv[1:]
j = json.load(open(p))
for v in j["videos"]:
    v["file_names"] = [f.replace(src, dst, 1)[:-4] + ".png" for f in v["file_names"]]
json.dump(j, open(p, "w"))
EOF
    fi
    log "fold $k: $(wc -w <<< "$train") training views -> $o"
    (cd "$HOIST_DIR" && FINETUNE_ALL_JSON="$d/train.json" EARLY_STOP_PATIENCE=800 EARLY_STOP_MIN_DELTA=0.01 \
        EARLY_STOP_WARMUP=300 EARLY_STOP_CHECK_PERIOD=50 \
        python "$SRC/train_finetune.py" --config-file "$CONFIG" --num-gpus 1 \
        OUTPUT_DIR "$o" DATALOADER.NUM_WORKERS 3 ${EXTRA_OPTS:-} > "$o/train.log" 2>&1) &
    local pid=$!
    # a run that stopped early can hang in its dataloader workers after the last checkpoint;
    # end it once the log reports the total training time. Keep the two newest checkpoints.
    while kill -0 "$pid" 2> /dev/null; do
        ls -t "$o"/model_0*.pth 2> /dev/null | tail -n +3 | xargs -r rm -f || true
        if grep -q "Total training time" "$o/train.log" 2> /dev/null; then
            sleep 30; pkill -P "$pid" 2> /dev/null || true; kill "$pid" 2> /dev/null || true
        fi
        sleep 30
    done
    if [ -f "$o/last_checkpoint" ]; then
        echo "    fold $k: $o/$(cat "$o/last_checkpoint")"
    else
        echo "    fold $k: no checkpoint, see $o/train.log" >&2; return 1
    fi
}

[ "$FOLDS" = all ] && FOLDS=all
pids=()
for k in $FOLDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
    train_fold "$k" & pids+=($!)
    sleep 60
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
log "done: $OUT"
exit $fail
