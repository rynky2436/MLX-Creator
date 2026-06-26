#!/bin/bash
# First-run setup for MLX Creator: install code, build the MLX venv (no torch),
# download the base model, then launch the app.
#   firstrun.sh "<APP_SUPPORT>" "<BUNDLE_APP>"
set -e
APP_SUPPORT="$1"
BUNDLE_APP="$2"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "============================================================"
echo "  MLX Creator — first-time setup"
echo "  (one time; builds the app environment. ~1-2 min.)"
echo "============================================================"

mkdir -p "$APP_SUPPORT"
echo "→ Installing app files…"
/usr/bin/rsync -a --exclude '.git' "$BUNDLE_APP"/ "$APP_SUPPORT"/

if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv (Python tool manager)…"
  /usr/bin/curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$APP_SUPPORT"
echo "→ Building Python environment (MLX, no PyTorch)…"
./setup_venv.sh

echo ""
echo "✓ Setup complete — launching MLX Creator."
echo "  Pick which models to download (image / music / video) in the app."
/usr/bin/open -a "MLX Creator"
echo "(You can close this window.)"
