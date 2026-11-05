#!/usr/bin/env bash
# Toggle "CVAT online" on the annotation machine (Ubuntu with GNOME):
#   first call:  disable idle suspend, start the CVAT stack, start the ngrok tunnel
#   second call: stop ngrok, stop the stack, restore the suspend settings
#
#     CVAT_DIR=~/cvat NGROK_URL=https://my-name.ngrok-free.dev bash run_web.sh
#
# Without NGROK_URL only the stack is started (local use). The compose files are those of
# docs/7_cvat.md; set COMPOSE_FILE in CVAT's .env to change them.
set -euo pipefail

CVAT_DIR="${CVAT_DIR:?set CVAT_DIR to the CVAT checkout}"
NGROK_BIN="${NGROK_BIN:-$(command -v ngrok || echo "$HOME/.local/bin/ngrok")}"
NGROK_URL="${NGROK_URL:-}"
CVAT_PORT="${CVAT_PORT:-8181}"

STATE_DIR="$HOME/.local/state/cvat-hosting"
AC_FILE="$STATE_DIR/ac_value"
BATTERY_FILE="$STATE_DIR/battery_value"
NGROK_PID_FILE="$STATE_DIR/ngrok.pid"
NGROK_LOG_FILE="$STATE_DIR/ngrok.log"
GS_SCHEMA="org.gnome.settings-daemon.plugins.power"
mkdir -p "$STATE_DIR"

start() {
  echo "== saving suspend settings, disabling idle suspend (the screen still turns off)"
  gsettings get "$GS_SCHEMA" sleep-inactive-ac-type | tr -d "'" > "$AC_FILE"
  gsettings get "$GS_SCHEMA" sleep-inactive-battery-type | tr -d "'" > "$BATTERY_FILE"
  gsettings set "$GS_SCHEMA" sleep-inactive-ac-type 'nothing'
  gsettings set "$GS_SCHEMA" sleep-inactive-battery-type 'nothing'

  echo "== starting CVAT"
  (cd "$CVAT_DIR" && docker compose up -d)

  if [ -n "$NGROK_URL" ]; then
    echo "== starting ngrok -> $NGROK_URL"
    nohup "$NGROK_BIN" http --url="$NGROK_URL" "$CVAT_PORT" > "$NGROK_LOG_FILE" 2>&1 &
    echo $! > "$NGROK_PID_FILE"
  fi
  echo "running. Call this script again to stop everything."
}

stop() {
  if [ -f "$NGROK_PID_FILE" ]; then
    echo "== stopping ngrok"
    kill "$(cat "$NGROK_PID_FILE")" 2> /dev/null || true
    rm -f "$NGROK_PID_FILE"
  fi
  echo "== stopping CVAT (data is kept in the docker volumes)"
  (cd "$CVAT_DIR" && docker compose down)

  echo "== restoring suspend settings"
  if [ -f "$AC_FILE" ] && [ -f "$BATTERY_FILE" ]; then
    gsettings set "$GS_SCHEMA" sleep-inactive-ac-type "$(cat "$AC_FILE")"
    gsettings set "$GS_SCHEMA" sleep-inactive-battery-type "$(cat "$BATTERY_FILE")"
    rm -f "$AC_FILE" "$BATTERY_FILE"
  else
    gsettings set "$GS_SCHEMA" sleep-inactive-ac-type 'suspend'
    gsettings set "$GS_SCHEMA" sleep-inactive-battery-type 'suspend'
  fi
  echo "stopped."
}

# saved state present = currently running
if [ -f "$NGROK_PID_FILE" ] || [ -f "$AC_FILE" ]; then stop; else start; fi
