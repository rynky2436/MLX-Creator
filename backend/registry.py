"""Installed-model registry. Scans models/ and tags each folder by modality /
engine / role so the UI tabs can list selectable models and the engines know
how to load them. Each model folder gets a self-describing mlxstudio.json.

Companion folders (LM planners, tokenizers) are role="companion" and never
appear as selectable models.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MANIFEST = "mlxstudio.json"

# Seed known folders from this build (so we don't depend on heuristics for them).
_SEED = {
    "FLUX.1-schnell": {"modality": "image", "engine": "flux", "arch": "schnell",
                        "display": "FLUX.1 schnell", "role": "model"},
    "FLUX.1-dev": {"modality": "image", "engine": "flux", "arch": "dev",
                   "display": "FLUX.1 dev", "role": "model"},
    "SD3.5-large": {"modality": "image", "engine": "sd35",
                    "display": "SD3.5 Large", "role": "model"},
    "SD3.5-large-4bit": {"modality": "image", "engine": "sd35",
                         "display": "SD3.5 Large (4-bit)", "role": "model"},
    "SD3-encoders": {"role": "companion"},
    "Qwen-Image-2512-4bit": {"modality": "image", "engine": "qwen",
                             "display": "Qwen-Image (4-bit)", "role": "model"},
    "Qwen-Image-2512-3bit": {"modality": "image", "engine": "qwen",
                             "display": "Qwen-Image (3-bit)", "role": "model"},
    "Qwen-Image-2512-8bit": {"modality": "image", "engine": "qwen",
                             "display": "Qwen-Image (8-bit)", "role": "model"},
    "ACE-Step1.5-MLX": {"modality": "audio", "engine": "ace_step",
                        "display": "ACE-Step 1.5", "role": "model"},
    "Wan2.2-TI2V-5B-MLX": {"modality": "video", "engine": "wan",
                           "display": "Wan 2.2 TI2V-5B", "role": "model"},
    "acestep-5Hz-lm-0.6B": {"role": "companion"},
    "acestep-5Hz-lm-4B": {"role": "companion"},
    "umt5-xxl-tokenizer": {"role": "companion"},
}


def _detect(d: Path) -> dict:
    """Infer a manifest from folder layout when none is seeded/present."""
    files = {p.name for p in d.iterdir() if p.is_file()}
    has_flux = any(f.startswith("flux1-") and f.endswith(".safetensors") for f in files) \
        and "ae.safetensors" in files
    if has_flux:
        arch = "dev" if any("dev" in f for f in files) else "schnell"
        return {"modality": "image", "engine": "flux", "arch": arch,
                "display": d.name, "role": "model"}
    cfg = d / "config.json"
    model_type = ""
    if cfg.exists():
        try:
            model_type = (json.loads(cfg.read_text()).get("model_type") or "").lower()
        except Exception:
            pass
    if model_type == "acestep":
        return {"modality": "audio", "engine": "ace_step", "display": d.name, "role": "model"}
    # Wan layout: dit + vae + t5 encoder
    if {"model.safetensors", "vae.safetensors", "t5_encoder.safetensors"} <= files:
        return {"modality": "video", "engine": "wan", "display": d.name, "role": "model"}
    # tokenizer-only or LM planner -> companion
    return {"role": "companion"}


def ensure_manifests() -> None:
    if not MODELS.exists():
        return
    for d in MODELS.iterdir():
        if not d.is_dir():
            continue
        mf = d / MANIFEST
        # The curated seed is authoritative — (re)write seeded folders so stale
        # auto-detected manifests can't shadow a known model.
        if d.name in _SEED:
            meta = dict(_SEED[d.name])
        elif mf.exists():
            continue
        else:
            meta = _detect(d)
        meta.setdefault("display", d.name)
        try:
            mf.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass


def _read(d: Path) -> dict:
    if d.name in _SEED:                      # seed wins over on-disk manifest
        return _SEED[d.name]
    mf = d / MANIFEST
    if mf.exists():
        try:
            return json.loads(mf.read_text())
        except Exception:
            pass
    return _detect(d)


def list_installed(modality: str | None = None) -> list[dict]:
    ensure_manifests()
    out = []
    if not MODELS.exists():
        return out
    for d in sorted(MODELS.iterdir()):
        if not d.is_dir():
            continue
        meta = _read(d)
        if meta.get("role") != "model":
            continue
        if modality and meta.get("modality") != modality:
            continue
        out.append({"id": d.name, "dir": str(d), **meta})
    return out


def get_dir(model_id: str) -> Path | None:
    d = MODELS / model_id
    return d if d.is_dir() else None
