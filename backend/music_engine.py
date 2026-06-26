"""Torch-free MLX music engine — ACE-Step 1.5 via mlx-audio.

Pure MLX: DiT + VAE + 5Hz LM planner all run through mlx / mlx-lm. No torch.
Weights load from the app-local models/ folder (download once, offline forever).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are all local

import mlx.core as mx  # noqa: E402
import scipy.io.wavfile as wavfile  # noqa: E402

# Point the 5Hz LM "thinking" planner(s) at local folders (download once).
from mlx_audio.tts.models.ace_step.lm import LMConfig, ACEStepLM  # noqa: E402

_PLANNER_DIRS = {
    "0.6B": MODELS / "acestep-5Hz-lm-0.6B",
    "4B": MODELS / "acestep-5Hz-lm-4B",
}
for _size, _dir in _PLANNER_DIRS.items():
    if _dir.exists():
        LMConfig._STANDALONE_MODEL_IDS[_size] = str(_dir)

# The planner computes chain-of-thought metadata (bpm/key/genre/structure) but
# only prints it. Wrap generate_audio_codes to capture it for the UI.
_LAST_META: dict = {}
_orig_gac = ACEStepLM.generate_audio_codes


def _gac_capture(self, *a, **k):
    out = _orig_gac(self, *a, **k)
    try:
        _LAST_META["meta"] = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    except Exception:
        _LAST_META["meta"] = None
    return out


ACEStepLM.generate_audio_codes = _gac_capture

from mlx_audio.tts import load as _load  # noqa: E402

ACESTEP_DIR = MODELS / "ACE-Step1.5-MLX"  # default
_MODEL = None
_MODEL_ID = None


def get_model(model_id: str = "ACE-Step1.5-MLX"):
    global _MODEL, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    _MODEL = _load(str(MODELS / model_id))
    _MODEL_ID = model_id
    return _MODEL


def unload():
    """Free the cached model (on model switch / shutdown)."""
    global _MODEL, _MODEL_ID
    _MODEL = None
    _MODEL_ID = None
    import gc
    gc.collect()


def model_status() -> dict:
    return {
        "acestep": {
            "present": (ACESTEP_DIR / "model.safetensors").exists(),
            "planners": {s: d.exists() for s, d in _PLANNER_DIRS.items()},
            "dir": str(ACESTEP_DIR),
        }
    }


def generate(
    *,
    prompt: str,
    lyrics: str = "",
    duration: float = 30.0,
    steps: int = 20,
    guidance: float = 1.0,
    shift: float = 3.0,
    vocal_language: str = "unknown",
    seed: Optional[int] = None,
    lm_size: str = "0.6B",
    model_id: str = "ACE-Step1.5-MLX",
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    if lm_size not in _PLANNER_DIRS or not _PLANNER_DIRS[lm_size].exists():
        lm_size = "0.6B"  # fall back if the chosen planner isn't downloaded
    if on_stage:
        on_stage("loading model")
    t_load0 = time.time()
    model = get_model(model_id)
    load_s = time.time() - t_load0

    _LAST_META.pop("meta", None)
    if on_stage:
        on_stage(f"thinking ({lm_size}) + diffusion")
    t0 = time.time()
    last = None
    for result in model.generate(
        text=prompt,
        lyrics=lyrics or "",
        duration=float(duration),
        num_steps=int(steps),
        guidance_scale=float(guidance),
        shift=float(shift),
        vocal_language=vocal_language or "unknown",
        seed=seed,
        use_lm=True,
        lm_model_size=lm_size,
        verbose=False,
    ):
        last = result
    gen_s = time.time() - t0
    meta = _LAST_META.get("meta") or {}

    if on_stage:
        on_stage("writing wav")
    audio = np.array(last.audio.astype(mx.float32))
    if audio.ndim == 2 and audio.shape[0] == 2:  # [2, samples] -> [samples, 2]
        audio = audio.T
    audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"acestep_{stamp}_{int(t0)}.wav"
    out_path = OUTPUTS / fname
    wavfile.write(str(out_path), int(last.sample_rate), audio_i16)

    peak_gb = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()

    return {
        "filename": fname,
        "path": str(out_path),
        "prompt": prompt,
        "lyrics": lyrics,
        "duration": duration,
        "steps": steps,
        "guidance": guidance,
        "vocal_language": vocal_language,
        "seed": seed if seed is not None else -1,
        "model": model_id,
        "lm_size": lm_size,
        "sample_rate": int(last.sample_rate),
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 2),
        "rtf": round(gen_s / max(duration, 0.1), 2),
        "peak_gb": round(peak_gb, 2),
        "thinking": {k: meta.get(k) for k in
                     ("bpm", "keyscale", "genres", "timesignature", "caption")
                     if isinstance(meta, dict) and meta.get(k) is not None},
    }
