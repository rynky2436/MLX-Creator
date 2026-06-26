#!/usr/bin/env bash
# Launch MLX Creator (torch-free MLX generative-media studio)
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_OFFLINE=1   # all weights are local in models/; never phone home
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8200}"
echo "MLX Creator → http://$HOST:$PORT"
exec uvicorn app:app --app-dir backend --host "$HOST" --port "$PORT" \
  --timeout-graceful-shutdown 2 "$@"
