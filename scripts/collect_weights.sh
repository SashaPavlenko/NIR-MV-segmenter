#!/bin/bash
# Put trained checkpoints into weights/ under the names the scripts expect:
#     weights/combo_nir/fold<k>.pth, weights/combo_depth/fold<k>.pth
#
#     bash scripts/collect_weights.sh <checkpoint root> [--link]
#
# <checkpoint root> holds <model>/fold<k>/ training folders (as written by train_lopo.sh:
# outputs/train), each with a last_checkpoint file that names the checkpoint to use.
# --link makes symbolic links instead of copies (saves disk space on the same machine).
set -euo pipefail
source "$(dirname "$0")/common.sh"
ROOT="${1:?usage: collect_weights.sh <checkpoint root> [--link]}"
MODE=copy; [ "${2:-}" = "--link" ] && MODE=link
n=0
for m in combo_nir combo_depth; do
    for k in 1 2 3 4 5 all; do
        d="$ROOT/$m/fold$k"
        [ -f "$d/last_checkpoint" ] || continue
        src="$d/$(cat "$d/last_checkpoint")"
        dst="$WEIGHTS_DIR/$m/fold$k.pth"
        mkdir -p "$(dirname "$dst")"
        if [ "$MODE" = link ]; then ln -sfn "$(realpath "$src")" "$dst"; else cp "$src" "$dst"; fi
        echo "$dst <- $src"
        n=$((n + 1))
    done
done
echo "$n checkpoints ($MODE)"
