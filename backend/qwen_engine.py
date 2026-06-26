"""Torch-free Qwen-Image engine via vendored mflux.

mflux's inference is pure-MLX; torch only appears in the weight loader's
original-checkpoint path + a VL processor type-check (both made lazy in the
vendored copy). We load pre-converted MLX weights from models/, so torch is
never touched. Quantization level is auto-detected from the weights' metadata.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "vendor" / "mflux" / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage  # noqa: E402
from mflux.models.common.config import ModelConfig  # noqa: E402

_MODEL = None
_MODEL_ID = None


def model_status() -> dict:
    return {
        "qwen": {
            mid: (MODELS / mid / "transformer").exists()
            for mid in ("Qwen-Image-2512-4bit", "Qwen-Image-2512-3bit", "Qwen-Image-2512-8bit")
        }
    }


def _round16(n: int) -> int:
    return ((n + 15) // 16) * 16


def get_model(model_id: str):
    global _MODEL, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    _MODEL = None
    import gc
    gc.collect()
    # quantize=None -> mflux resolves the stored quant level from the weights' metadata
    _MODEL = QwenImage(
        quantize=None,
        model_path=str(MODELS / model_id),
        model_config=ModelConfig.qwen_image(),
    )
    _MODEL_ID = model_id
    return _MODEL


def generate(
    *,
    prompt: str,
    model: str = "Qwen-Image-2512-4bit",
    steps: int = 20,
    guidance: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    on_step: Optional[Callable[[int, int], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the (mflux) generation in a SUBPROCESS.

    mflux's Qwen VAE decode aborts (C++ std::runtime_error) when run off the
    main thread — and our API worker is a background thread. A subprocess gets
    its own main thread (weights mmap in <1s) and also isolates any crash from
    the server.
    """
    import json
    import subprocess
    if on_stage:
        on_stage("loading model")
    payload = json.dumps({
        "prompt": prompt, "model": model, "steps": steps, "guidance": guidance,
        "width": width, "height": height, "seed": seed,
        "negative_prompt": negative_prompt,
    })
    if on_stage:
        on_stage("denoising (subprocess)")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "qwen_runner.py")],
        input=payload, capture_output=True, text=True, env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError("Qwen subprocess failed:\n" + (proc.stderr or proc.stdout)[-800:])


def _generate_core(
    *,
    prompt: str,
    model: str = "Qwen-Image-2512-4bit",
    steps: int = 20,
    guidance: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    on_step: Optional[Callable[[int, int], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    height, width = _round16(height), _round16(width)
    used_seed = seed if seed is not None else int(time.time()) % (2**31)

    if on_stage:
        on_stage("loading model")
    t_load0 = time.time()
    model_obj = get_model(model)
    load_s = time.time() - t_load0

    if on_stage:
        on_stage("denoising")
    t0 = time.time()
    result = model_obj.generate_image(
        seed=int(used_seed),
        prompt=prompt,
        num_inference_steps=int(steps),
        height=height,
        width=width,
        guidance=float(guidance),
        negative_prompt=negative_prompt or None,
    )
    gen_s = time.time() - t0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"qwen_{stamp}_{int(t0)}.png"
    out_path = OUTPUTS / fname
    result.image.save(out_path)

    peak_gb = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()
    return {
        "filename": fname, "path": str(out_path), "prompt": prompt,
        "model": model, "steps": steps, "guidance": guidance,
        "width": width, "height": height, "seed": used_seed,
        "load_s": round(load_s, 2), "gen_s": round(gen_s, 2),
        "peak_gb": round(peak_gb, 2),
    }
