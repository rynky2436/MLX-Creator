#!/usr/bin/env bash
# Recreate the MLX Creator Python environment (torch-free, MLX-native).
# Safe to run in a fresh clone of this app folder. Requires: uv, network.
set -e
cd "$(dirname "$0")"
echo "Creating venv (.venv) with Python 3.12…"
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "→ image (Flux, torch-free) + API server deps"
uv pip install -r vendor/mlx-examples/flux/requirements.txt \
  fastapi "uvicorn[standard]" python-multipart

echo "→ music (ACE-Step 1.5 via mlx-audio, torch-free pc/add-ace branch)"
uv pip install "git+https://github.com/shreyaskarnik/mlx-audio.git@pc/add-ace"

echo "→ video (Wan 2.2 via mlx-video; --no-deps to skip librosa/numba build break)"
uv pip install --no-deps "git+https://github.com/Blaizzy/mlx-video.git"
uv pip install tqdm "opencv-python>=4.12.0.88" "Pillow>=10.3.0" \
  imageio imageio-ffmpeg ftfy "rich>=14.2.0" mlx-vlm

echo "→ extra image engines (vendored, torch-free): SD3/3.5 (DiffusionKit) + Qwen-Image (mflux)"
# DiffusionKit needs beartype; mflux needs platformdirs/piexif/toml/matplotlib.
# (The engines themselves are vendored under vendor/ — only their small deps here.)
uv pip install beartype platformdirs piexif toml matplotlib

echo "→ verifying torch-free…"
python -c "import importlib.util as u; assert not u.find_spec('torch'), 'TORCH PRESENT!'; print('OK: no torch')"
echo "Done. Launch with ./run.sh"
