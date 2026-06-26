#!/bin/bash
# MLX Creator.app launcher. On first run, sets up the env + downloads the base
# model (in Terminal, for visible progress). On later runs, starts the server
# and opens the UI; quitting the app stops the server.
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
RES="$CONTENTS/Resources"
BUNDLE_APP="$RES/app"
APP_SUPPORT="$HOME/Library/Application Support/MLX Creator"
DEV="$HOME/mlx-studio"
PORT="${PORT:-8200}"

# "Ready" = the venv exists. Models are NOT required to start — the app's
# welcome screen lets the user pick/download base models on first run.
ready() { [ -x "$1/.venv/bin/python" ]; }

# Prefer an existing install (Application Support, then a dev checkout) to avoid
# re-downloading; otherwise do first-run setup.
if ready "$APP_SUPPORT"; then
  RUNDIR="$APP_SUPPORT"
elif ready "$DEV"; then
  RUNDIR="$DEV"
else
  osascript >/dev/null 2>&1 <<OSA
tell application "Terminal"
  activate
  do script "'$RES/firstrun.sh' '$APP_SUPPORT' '$BUNDLE_APP'"
end tell
OSA
  exit 0
fi

# Normal launch.
cd "$RUNDIR" || exit 1
export PORT

# Port guard: if a server is already running on this port, just open the UI
# instead of starting a duplicate (which would fail to bind and break things).
if /usr/bin/curl -fsS "http://127.0.0.1:$PORT/api/installed" >/dev/null 2>&1; then
  /usr/bin/open "http://127.0.0.1:$PORT"
  exit 0
fi

./run.sh > "$RUNDIR/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT INT TERM
for i in $(seq 1 90); do
  /usr/bin/curl -fsS "http://127.0.0.1:$PORT/api/installed" >/dev/null 2>&1 && break
  sleep 1
done
/usr/bin/open "http://127.0.0.1:$PORT"
wait $SRV
