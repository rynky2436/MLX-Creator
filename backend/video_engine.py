"""Torch-free MLX video engine — Wan 2.2 TI2V-5B via mlx-video.

Pure MLX (DiT + Wan2.2 VAE + T5 text encoder). No torch on the inference path
(mlx-video's only torch import is its offline convert.py, never loaded here).
Weights load from the app-local models/ folder. Writes an mp4 via imageio.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
import config
MODELS = config.models_dir()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mlx_video.models.wan_2.generate import generate_video  # noqa: E402

WAN_DIR = MODELS / "Wan2.2-TI2V-5B-MLX"
_UMT5_TOK = MODELS / "umt5-xxl-tokenizer"

FPS = 24  # Wan 2.2 TI2V-5B native sample_fps


def snap_frames(n: int) -> int:
    """Snap a frame count to the nearest valid 4n+1 (model constraint)."""
    k = max(1, round((n - 1) / 4))
    return k * 4 + 1


def seconds_to_frames(seconds: float) -> int:
    return snap_frames(round(seconds * FPS))

# mlx-video hardcodes AutoTokenizer.from_pretrained("google/umt5-xxl") (a Hub
# fetch). Redirect that single call to the local tokenizer folder so we stay
# fully offline / all-local.
import transformers  # noqa: E402

_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained.__func__


def _local_tokenizer(cls, name, *a, **k):
    if name == "google/umt5-xxl" and _UMT5_TOK.exists():
        name = str(_UMT5_TOK)
    return _orig_from_pretrained(cls, name, *a, **k)


transformers.AutoTokenizer.from_pretrained = classmethod(_local_tokenizer)


def model_status() -> dict:
    return {
        "wan22": {
            "present": (WAN_DIR / "model.safetensors").exists(),
            "dir": str(WAN_DIR),
        }
    }


def _round_to(n: int, mult: int) -> int:
    return max(mult, (n // mult) * mult)


def generate(
    *,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 704,
    height: int = 480,
    num_frames: int = 49,
    duration: Optional[float] = None,
    steps: int = 20,
    guidance: Optional[float] = None,
    shift: Optional[float] = None,
    seed: int = -1,
    image: Optional[str] = None,
    model_id: str = "Wan2.2-TI2V-5B-MLX",
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run Wan generation in a SUBPROCESS. mlx-video holds the model inside its
    own internals (no handle for us to unload), so the only reliable way to free
    its ~9GB is to let a subprocess exit — the OS reclaims everything."""
    import json
    import subprocess
    import sys
    if on_stage:
        on_stage("loading model")
    payload = json.dumps({
        "prompt": prompt, "negative_prompt": negative_prompt,
        "width": width, "height": height, "num_frames": num_frames,
        "duration": duration, "steps": steps, "guidance": guidance,
        "shift": shift, "seed": seed, "image": image, "model_id": model_id,
    })
    if on_stage:
        on_stage("generating video (subprocess)")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "video_runner.py")],
        input=payload, capture_output=True, text=True, env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError("Video subprocess failed:\n" + (proc.stderr or proc.stdout)[-800:])


def _generate_core(
    *,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 704,
    height: int = 480,
    num_frames: int = 49,
    duration: Optional[float] = None,
    steps: int = 20,
    guidance: Optional[float] = None,
    shift: Optional[float] = None,
    seed: int = -1,
    image: Optional[str] = None,
    model_id: str = "Wan2.2-TI2V-5B-MLX",
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    model_dir = MODELS / model_id
    if not (model_dir / "model.safetensors").exists():
        model_dir = WAN_DIR  # fall back to default
    # Duration (seconds) is the primary control; convert to a valid 4n+1 frame
    # count at the model's native fps. Falls back to num_frames if no duration.
    num_frames = seconds_to_frames(duration) if duration else snap_frames(num_frames)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    fname = f"wan22_{stamp}_{int(t0)}.mp4"
    out_path = OUTPUTS / fname

    if on_stage:
        on_stage("loading model")

    # mlx-video runs the whole pipeline and writes the mp4 itself.
    if on_stage:
        on_stage("generating video")
    generate_video(
        model_dir=str(model_dir),
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        steps=int(steps),
        guide_scale=guidance,
        shift=shift,
        seed=int(seed),
        output_path=str(out_path),
        scheduler="unipc",
    )
    gen_s = time.time() - t0

    import mlx.core as mx
    peak_gb = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()

    size_b = out_path.stat().st_size if out_path.exists() else 0
    return {
        "filename": fname,
        "path": str(out_path),
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": FPS,
        "duration_s": round(num_frames / FPS, 2),
        "steps": steps,
        "guidance": guidance if guidance is not None else "config",
        "seed": seed,
        "mode": "i2v" if image else "t2v",
        "gen_s": round(gen_s, 1),
        "peak_gb": round(peak_gb, 2),
        "size_mb": round(size_b / 1e6, 2),
    }
