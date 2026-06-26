"""Torch-free MLX Flux engine — wraps ml-explore/mlx-examples/flux.

Exposes a single generate() with a per-step progress callback so the API
layer can stream live progress over a websocket. The heavy FluxPipeline is
loaded once and cached, keyed by (model, quantize).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

# vendor the mlx-examples flux package
ROOT = Path(__file__).resolve().parent.parent
FLUX_DIR = ROOT / "vendor" / "mlx-examples" / "flux"
sys.path.insert(0, str(FLUX_DIR))

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from flux import FluxPipeline  # noqa: E402
import flux.utils as _flux_utils  # noqa: E402

# Load weights from the app-local models/ folder (download once, forever local).
# Each model lives in models/<NAME>/ with the BFL file layout. We intercept the
# loader's hf_hub_download so any file present locally is used directly — the
# network is only ever touched if a file is genuinely missing.
import config
MODELS_DIR = config.models_dir()
# Default known dirs + an "active" dir set per generation so any installed Flux
# model folder can be loaded. The loader's hf_hub_download is intercepted to use
# local files first (active dir wins), so the network is never touched.
_LOCAL_DIRS = {
    "flux-schnell": MODELS_DIR / "FLUX.1-schnell",
    "flux-dev": MODELS_DIR / "FLUX.1-dev",
}
_ACTIVE_DIR: Optional[Path] = None
_orig_hf_download = _flux_utils.hf_hub_download


def _local_first_download(repo_id, filename, *args, **kwargs):
    search = ([_ACTIVE_DIR] if _ACTIVE_DIR else []) + list(_LOCAL_DIRS.values())
    for d in search:
        p = d / filename
        if p.exists():
            return str(p)
    return _orig_hf_download(repo_id, filename, *args, **kwargs)


_flux_utils.hf_hub_download = _local_first_download


def model_status() -> dict:
    """Report which local models are present (for the API / UI)."""
    out = {}
    for name, d in _LOCAL_DIRS.items():
        flow = d / ("flux1-%s.safetensors" % name.split("-")[1])
        out[name] = {"present": flow.exists(), "dir": str(d)}
    return out

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

_PIPE: Optional[FluxPipeline] = None
_PIPE_KEY: Optional[tuple] = None


def _quant_predicate(_path, m):
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0


def _round16(n: int) -> int:
    return ((n + 15) // 16) * 16


def get_pipeline(model_id: str, arch: str, quantize: bool) -> FluxPipeline:
    """Load (or reuse) a FluxPipeline for an installed model folder.

    model_id: folder under models/ (e.g. 'FLUX.1-schnell')
    arch:     'schnell' or 'dev' — selects the mlx-examples architecture/config
    """
    global _PIPE, _PIPE_KEY, _ACTIVE_DIR
    key = (model_id, quantize)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    # drop any previous pipeline before loading a new one
    _PIPE = None
    gc.collect()

    _ACTIVE_DIR = MODELS_DIR / model_id   # resolver pulls weights from here
    name = "flux-" + arch                 # architecture (schnell|dev)
    pipe = FluxPipeline(name, t5_padding=True)
    if quantize:
        nn.quantize(pipe.flow, class_predicate=_quant_predicate)
        nn.quantize(pipe.t5, class_predicate=_quant_predicate)
        nn.quantize(pipe.clip, class_predicate=_quant_predicate)

    _PIPE = pipe
    _PIPE_KEY = key
    return pipe


def unload():
    """Free the cached pipeline (on model switch / shutdown)."""
    global _PIPE, _PIPE_KEY
    _PIPE = None
    _PIPE_KEY = None
    gc.collect()


def _resolve_arch(model_id: str) -> str:
    """schnell|dev for an installed Flux folder (from manifest, else name)."""
    try:
        import registry
        for m in registry.list_installed("image"):
            if m["id"] == model_id and m.get("arch"):
                return m["arch"]
    except Exception:
        pass
    return "dev" if "dev" in model_id.lower() else "schnell"


def generate(
    *,
    prompt: str,
    model: str = "FLUX.1-schnell",  # installed folder id (legacy: 'schnell'/'dev')
    steps: Optional[int] = None,
    guidance: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    quantize: bool = True,
    on_step: Optional[Callable[[int, int], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate one image. Returns dict with filename, path, timings, params."""
    # legacy aliases
    model_id = {"schnell": "FLUX.1-schnell", "dev": "FLUX.1-dev"}.get(model, model)
    arch = _resolve_arch(model_id)
    if steps is None:
        steps = 4 if arch == "schnell" else 25
    # schnell is tuned for cfg/guidance 1; it ignores guidance but keep sane default
    height, width = _round16(height), _round16(width)
    latent_size = (height // 8, width // 8)

    if on_stage:
        on_stage("loading model")
    t_load0 = time.time()
    pipe = get_pipeline(model_id, arch, quantize)
    load_s = time.time() - t_load0

    if on_stage:
        on_stage("encoding prompt")
    t_gen0 = time.time()

    latents = pipe.generate_latents(
        prompt,
        n_images=1,
        num_steps=steps,
        guidance=guidance,
        latent_size=latent_size,
        seed=seed,
    )

    # first yield = conditioning; evaluate then free text encoders
    conditioning = next(latents)
    mx.eval(conditioning)
    pipe.reload_text_encoders()

    if on_stage:
        on_stage("denoising")
    x_t = None
    for i, x_t in enumerate(latents):
        mx.eval(x_t)
        if on_step:
            on_step(i + 1, steps)

    if on_stage:
        on_stage("decoding")
    img = pipe.decode(x_t, latent_size)
    mx.eval(img)
    gen_s = time.time() - t_gen0

    # to uint8 PNG
    arr = (np.array(img[0]) * 255).astype(np.uint8)
    used_seed = seed if seed is not None else -1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{arch}_{stamp}_{int(t_gen0)}.png"
    out_path = OUTPUTS / fname
    Image.fromarray(arr).save(out_path)

    peak_gb = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()

    return {
        "filename": fname,
        "path": str(out_path),
        "prompt": prompt,
        "model": model_id,
        "arch": arch,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "seed": used_seed,
        "quantize": quantize,
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 2),
        "peak_gb": round(peak_gb, 2),
    }
