"""Torch-free SD3 / SD3.5 image engine via vendored DiffusionKit (MLX).

DiffusionKit's MLX inference code is pure-MLX; torch only lived in conversion +
a logger util. We vendor the mlx subpackage, stub argmaxtools, make one optional
import lazy, and load weights from the app-local models/ folder (offline).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# vendor paths: stub argmaxtools (no torch) + DiffusionKit source
sys.path.insert(0, str(ROOT / "vendor" / "argmaxtools_stub"))
sys.path.insert(0, str(ROOT / "vendor" / "DiffusionKit" / "python" / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402

# MLX 0.17 -> 0.31 drift: scaled_dot_product_attention dropped the old
# `memory_efficient_threshold` kwarg. Strip it so DiffusionKit's mmdit works.
_orig_sdpa = mx.fast.scaled_dot_product_attention


def _sdpa_compat(*a, **k):
    k.pop("memory_efficient_threshold", None)
    return _orig_sdpa(*a, **k)


mx.fast.scaled_dot_product_attention = _sdpa_compat

from diffusionkit.mlx import DiffusionPipeline  # noqa: E402
import diffusionkit.mlx.model_io as _mio  # noqa: E402

# Map DiffusionKit HF repo ids -> app-local folders (download once, load offline)
_REPO_LOCAL = {
    "argmaxinc/stable-diffusion": MODELS / "SD3-encoders",
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": MODELS / "SD3.5-large-4bit",
    "argmaxinc/mlx-stable-diffusion-3.5-large": MODELS / "SD3.5-large",
}
# installed folder id -> DiffusionKit model_version key
_MODEL_KEY = {
    "SD3.5-large-4bit": "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized",
    "SD3.5-large": "argmaxinc/mlx-stable-diffusion-3.5-large",
}

_orig_hf = _mio.hf_hub_download


def _local_first(repo_id, filename, *a, **k):
    d = _REPO_LOCAL.get(repo_id)
    if d:
        p = d / filename
        if p.exists():
            return str(p)
    return _orig_hf(repo_id, filename, *a, **k)


_mio.hf_hub_download = _local_first

# DiffusionKit hardcodes T5Config/AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
# (Hub fetches). Redirect to the local config/tokenizer so we stay offline.
_T5_LOCAL = MODELS / "SD3-encoders" / "t5-google-config"
import transformers  # noqa: E402

for _cls in (transformers.T5Config, transformers.AutoTokenizer):
    _orig = _cls.from_pretrained.__func__

    def _mk(orig):
        def _redir(cls, name, *a, **k):
            if name == "google/t5-v1_1-xxl" and _T5_LOCAL.exists():
                name = str(_T5_LOCAL)
            return orig(cls, name, *a, **k)
        return classmethod(_redir)

    _cls.from_pretrained = _mk(_orig)

_PIPE = None
_PIPE_KEY = None


def model_status() -> dict:
    return {
        "sd35": {
            "encoders": (MODELS / "SD3-encoders" / "t5" / "t5xxl.safetensors").exists(),
            "variants": {mid: (MODELS / mid).exists() for mid in _MODEL_KEY},
        }
    }


def _round16(n: int) -> int:
    return ((n + 15) // 16) * 16


def _resolve_key(model_id: str) -> str:
    """DiffusionKit model_version key for an installed folder.

    Seeded folders use _MODEL_KEY; browser-installed ones read the source repo
    (which IS the DiffusionKit key) from their manifest, and we register the
    folder so the loader pulls the mmdit locally.
    """
    if model_id in _MODEL_KEY:
        return _MODEL_KEY[model_id]
    import json
    mf = MODELS / model_id / "mlxstudio.json"
    if mf.exists():
        try:
            src = json.loads(mf.read_text()).get("source")
            if src:
                _REPO_LOCAL[src] = MODELS / model_id  # load mmdit from here
                return src
        except Exception:
            pass
    return model_id


def get_pipeline(model_id: str):
    global _PIPE, _PIPE_KEY
    if _PIPE is not None and _PIPE_KEY == model_id:
        return _PIPE
    _PIPE = None
    import gc
    gc.collect()
    key = _resolve_key(model_id)
    _PIPE = DiffusionPipeline(
        model_version=key, shift=3.0, use_t5=True,
        w16=True, a16=True, low_memory_mode=True,
    )
    _PIPE_KEY = model_id
    return _PIPE


def unload():
    """Free the cached pipeline (on model switch / shutdown)."""
    global _PIPE, _PIPE_KEY
    _PIPE = None
    _PIPE_KEY = None
    import gc
    gc.collect()


def generate(
    *,
    prompt: str,
    model: str = "SD3.5-large-4bit",
    steps: int = 28,
    guidance: float = 4.5,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    on_step: Optional[Callable[[int, int], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    height, width = _round16(height), _round16(width)
    latent_size = (height // 8, width // 8)

    # SD3/3.5 are not distilled — they require classifier-free guidance. With
    # guidance <= 1 DiffusionKit takes a batch-1 fast-path that crashes
    # (layer_norm weight becomes 2-D), so enforce a sane CFG minimum.
    if guidance is None or guidance <= 1:
        guidance = 4.5

    if on_stage:
        on_stage("loading model")
    t_load0 = time.time()
    pipe = get_pipeline(model)
    load_s = time.time() - t_load0

    if on_stage:
        on_stage("denoising")
    t0 = time.time()
    image, _log = pipe.generate_image(
        text=prompt,
        num_steps=int(steps),
        cfg_weight=float(guidance),
        negative_text=negative_prompt or "",
        latent_size=latent_size,
        seed=seed,
        verbose=False,
    )
    gen_s = time.time() - t0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"sd35_{stamp}_{int(t0)}.png"
    out_path = OUTPUTS / fname
    image.save(out_path)

    peak_gb = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()
    return {
        "filename": fname, "path": str(out_path), "prompt": prompt,
        "model": model, "steps": steps, "guidance": guidance,
        "width": width, "height": height, "seed": seed if seed is not None else -1,
        "load_s": round(load_s, 2), "gen_s": round(gen_s, 2),
        "peak_gb": round(peak_gb, 2),
    }
