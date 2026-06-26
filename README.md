# MLX Creator

Local generative-media studio for Apple Silicon — make **images, music, and video** from one app, fully on-device via [MLX](https://github.com/ml-explore/mlx). No PyTorch, no cloud, runs offline.

It's the "LM Studio" idea applied to generative media: a clean UI + a built-in model browser, with pluggable MLX engines under the hood.

## Engines

| Modality | Models | Runtime |
|---|---|---|
| Image | FLUX (schnell/dev), SD3 / SD3.5, Qwen-Image | mlx-examples · DiffusionKit · mflux |
| Music | ACE-Step 1.5 (incl. 4-bit; 0.6B/4B "thinking" planner) | mlx-audio |
| Video | Wan 2.2 TI2V-5B (text→video) | mlx-video |

A built-in **Models** tab browses HuggingFace (MLX-only) and Civitai and installs models straight into `models/`, auto-registering them in the right tab.

## Install & run

```bash
git clone https://github.com/rynky2436/MLX-Creator.git
cd MLX-Creator
./install.sh      # builds the MLX venv (no torch) + pulls the 3 base models
./run.sh          # serve at http://127.0.0.1:8200
```

`install.sh` downloads the three base models — **Flux** (image), **ACE-Step 1.5**
(music), **Wan 2.2** (video), ~72 GB total — into `models/` (git-ignored). It's
resumable; re-running skips anything already present. Grab more models anytime
from the in-app **Models** tab.

## Desktop app (.dmg)

`packaging/build_dmg.sh` builds **MLX Creator.app** + **MLX-Creator.dmg** (a
double-click app with a menu/Dock icon; first launch downloads the base models).

The app is **not notarized** (no paid Apple Developer cert), so the first time
you open it macOS Gatekeeper will block it. Open it once with either:

```bash
xattr -dr com.apple.quarantine "/Applications/MLX Creator.app"
```

or **right-click the app → Open** (then *System Settings ▸ Privacy & Security ▸
Open Anyway* if needed). After that it opens normally. The DMG includes a
`FIRST-LAUNCH.txt` with these steps.

## Layout

```
backend/    FastAPI server + per-modality MLX engines + model registry
frontend/   single-file web UI
vendor/     patched, torch-free copies of mlx-examples, DiffusionKit, mflux
models/     downloaded weights (not committed)
outputs/    generated images / audio / video (not committed)
```

## Notes

Vendored libraries are patched to keep their **MLX inference paths torch-free** (torch only ever lived in their weight-conversion/CLI code). See each `vendor/` subtree for the upstream project and its license.
