#!/bin/bash
# Write production_patched.py into a CVAT checkout: CVAT's own cvat/settings/production.py
# plus three lines that read CSRF_TRUSTED_ORIGINS from the environment (for docker-compose.remote.yml).
#
#     bash make_production_patch.sh /path/to/cvat
set -euo pipefail
CVAT="${1:?usage: make_production_patch.sh <cvat checkout>}"
SRC="$CVAT/cvat/settings/production.py"
[ -f "$SRC" ] || { echo "not found: $SRC" >&2; exit 1; }
{
    cat "$SRC"
    echo
    echo "import os as _os"
    echo 'CSRF_TRUSTED_ORIGINS = _os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")'
} > "$CVAT/production_patched.py"
echo "written: $CVAT/production_patched.py"
