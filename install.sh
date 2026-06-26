#!/usr/bin/env bash
# MLX Creator installer — run after cloning from GitHub.
#   git clone https://github.com/rynky2436/MLX-Creator.git
#   cd MLX-Creator && ./install.sh
# Builds the MLX (torch-free) environment and pulls the 3 base models
# (Flux · ACE-Step 1.5 · Wan 2.2). Re-running is safe / resumable.
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  MLX Creator — install"
echo "============================================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv (Python tool manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "→ Building Python environment (MLX, no PyTorch)…"
./setup_venv.sh

echo ""
echo "✓ Install complete. Launch with:  ./run.sh   → http://127.0.0.1:8200"
echo "  On first launch, the app's welcome screen lets you pick which models to"
echo "  download (image / music / video) — choose any, all, or none."
echo ""
echo "  (Prefer to pre-pull all 3 base models now? run: "
echo "     .venv/bin/python backend/download_base_models.py )"
