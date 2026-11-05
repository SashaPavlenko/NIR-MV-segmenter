#!/bin/bash
# Apply the thesis changes to a clean HOIST-Former checkout and install the model configs.
#
#     bash hoistformer_patch/apply_patch.sh third_party/HOISTFormer
#
# (scripts/setup.sh calls this.) The patch is `git diff` against upstream commit a229e83.
# files/ holds the same changes as whole files, for the case where the patch no longer
# applies to a newer upstream.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:?usage: apply_patch.sh <HOISTFormer checkout>}"

cd "$TARGET"
if git apply --check "$HERE/hoistformer_thesis.patch" 2> /dev/null; then
    git apply "$HERE/hoistformer_thesis.patch"
    echo "    patch applied"
elif git apply --reverse --check "$HERE/hoistformer_thesis.patch" 2> /dev/null; then
    echo "    patch already applied"
else
    echo "    patch does not apply (upstream changed?): compare with $HERE/files/ and copy by hand" >&2
    exit 1
fi
# new file, not part of the diff
cp "$HERE/files/mask2former_video/data_video/multiview_dataset_mapper.py" \
   mask2former_video/data_video/multiview_dataset_mapper.py
cp "$HERE"/configs/hoist/*.yaml configs/hoist/
echo "    configs installed: $(ls "$HERE"/configs/hoist | tr '\n' ' ')"
