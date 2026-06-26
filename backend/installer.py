"""Download an installed model from HuggingFace into the app-local models/
folder, with progress, then write a manifest so it shows up in the right tab.

Forces online for the duration of the download even though the app otherwise
runs with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
import config
MODELS = config.models_dir()

_IGNORE = ["*.md", ".gitattributes", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4"]

# Some engines need shared companion repos (downloaded once, reused).
# (repo, dest-folder, allow_patterns) — skipped if dest already populated.
_COMPANIONS = {
    "sd35": [
        ("argmaxinc/stable-diffusion", "SD3-encoders",
         ["clip_l/*", "clip_g/*", "t5/*", "tokenizer_l/*", "tokenizer_g/*"]),
        ("google/t5-v1_1-xxl", "SD3-encoders/t5-google-config",
         ["config.json", "spiece.model", "tokenizer_config.json",
          "special_tokens_map.json", "tokenizer.json"]),
    ],
}


# The 3 recommended base models, one per modality. Each may be several repos
# (primary + companions). Sizes are approximate (for the welcome-screen + bar).
BASE_RECIPES = {
    "image": {
        "display": "FLUX.1 schnell", "modality": "image", "engine": "flux",
        "arch": "schnell", "size_gb": 34, "ready": "flux1-schnell.safetensors",
        "repos": [("MLXCreator/MLXCreator-Flux-Schnell", "FLUX.1-schnell", None)],
    },
    "music": {
        "display": "ACE-Step 1.5", "modality": "audio", "engine": "ace_step",
        "size_gb": 14, "ready": "model.safetensors",
        "repos": [("MLXCreator/MLXCreator-ACEStep-1.5", "ACE-Step1.5-MLX", None),
                  ("MLXCreator/MLXCreator-ACEStep-Planner-0.6B", "acestep-5Hz-lm-0.6B", None)],
    },
    "video": {
        "display": "Wan 2.2 TI2V-5B", "modality": "video", "engine": "wan",
        "size_gb": 24, "ready": "model.safetensors",
        "repos": [("MLXCreator/MLXCreator-Wan2.2-TI2V-5B", "Wan2.2-TI2V-5B-MLX", None),
                  ("MLXCreator/MLXCreator-UMT5-Tokenizer", "umt5-xxl-tokenizer", None)],
    },
}


def base_models_status() -> list[dict]:
    """The 3 base models with install state — for the welcome screen."""
    out = []
    for key, rec in BASE_RECIPES.items():
        prim = MODELS / rec["repos"][0][1]
        out.append({
            "key": key, "display": rec["display"], "modality": rec["modality"],
            "engine": rec["engine"], "size_gb": rec["size_gb"],
            "installed": (prim / rec["ready"]).exists(),
        })
    return out


def download_base(base_key: str, on_progress=None, on_stage=None) -> dict:
    """Download one base recipe (all its repos) with cumulative progress."""
    from huggingface_hub import snapshot_download
    import huggingface_hub.constants as hc

    rec = BASE_RECIPES[base_key]
    dests = [MODELS / d for _, d, _ in rec["repos"]]
    for d in dests:
        d.mkdir(parents=True, exist_ok=True)
    total = rec["size_gb"] * 1e9

    prev = hc.HF_HUB_OFFLINE
    hc.HF_HUB_OFFLINE = False
    stop = {"f": False}

    def poll():
        while not stop["f"]:
            if on_progress:
                on_progress(min(0.99, sum(_dir_size(d) for d in dests) / total))
            time.sleep(1.0)

    if on_stage:
        on_stage("downloading")
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        for repo, dest, allow in rec["repos"]:
            snapshot_download(repo, local_dir=str(MODELS / dest), allow_patterns=allow,
                              ignore_patterns=_IGNORE, max_workers=8)
    finally:
        stop["f"] = True
        hc.HF_HUB_OFFLINE = prev

    prim = MODELS / rec["repos"][0][1]
    meta = {"modality": rec["modality"], "engine": rec["engine"],
            "display": rec["display"], "role": "model"}
    if rec.get("arch"):
        meta["arch"] = rec["arch"]
    (prim / "mlxstudio.json").write_text(json.dumps(meta, indent=2))
    if on_progress:
        on_progress(1.0)
    return {"id": rec["repos"][0][1], "modality": rec["modality"],
            "size_gb": round(sum(_dir_size(d) for d in dests) / 1e9, 2)}


def download_planner4b(on_progress=None, on_stage=None) -> dict:
    """Download the optional ACE-Step 4B 'thinking' planner into its companion folder."""
    from huggingface_hub import snapshot_download
    import huggingface_hub.constants as hc

    dest = MODELS / "acestep-5Hz-lm-4B"
    dest.mkdir(parents=True, exist_ok=True)
    total = 8 * 1e9
    prev = hc.HF_HUB_OFFLINE
    hc.HF_HUB_OFFLINE = False
    stop = {"f": False}

    def poll():
        while not stop["f"]:
            if on_progress:
                on_progress(min(0.99, _dir_size(dest) / total))
            time.sleep(1.0)

    if on_stage:
        on_stage("downloading")
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        snapshot_download("MLXCreator/MLXCreator-ACEStep-Planner-4B",
                          local_dir=str(dest), ignore_patterns=_IGNORE, max_workers=8)
    finally:
        stop["f"] = True
        hc.HF_HUB_OFFLINE = prev
    (dest / "mlxstudio.json").write_text(json.dumps(
        {"role": "companion", "display": "ACE-Step Planner 4B"}, indent=2))
    if on_progress:
        on_progress(1.0)
    return {"id": "acestep-5Hz-lm-4B", "size_gb": round(_dir_size(dest) / 1e9, 2)}


def _repo_size(repo: str) -> int:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings)
    except Exception:
        return 0


def _dir_size(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def download_hf(
    *,
    repo: str,
    modality: str,
    engine: str,
    arch: Optional[str] = None,
    display: Optional[str] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict:
    from huggingface_hub import snapshot_download
    import huggingface_hub.constants as hc

    name = repo.split("/")[-1]
    dest = MODELS / name
    dest.mkdir(parents=True, exist_ok=True)

    if on_stage:
        on_stage("sizing")
    total = _repo_size(repo)

    # force online just for the download
    prev_offline = hc.HF_HUB_OFFLINE
    hc.HF_HUB_OFFLINE = False

    # fetch shared companion repos this engine needs (once)
    for crepo, cdest, callow in _COMPANIONS.get(engine, []):
        cpath = MODELS / cdest
        if cpath.exists() and any(cpath.rglob("*.safetensors")) or \
           (cpath.exists() and any(cpath.iterdir())):
            continue
        if on_stage:
            on_stage(f"companion: {cdest}")
        cpath.mkdir(parents=True, exist_ok=True)
        snapshot_download(crepo, local_dir=str(cpath), allow_patterns=callow,
                          ignore_patterns=_IGNORE, max_workers=8)

    stop = {"f": False}

    def poll():
        while not stop["f"]:
            if on_progress and total:
                on_progress(min(0.99, _dir_size(dest) / total))
            time.sleep(1.0)

    if on_stage:
        on_stage("downloading")
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        snapshot_download(repo, local_dir=str(dest), ignore_patterns=_IGNORE, max_workers=8)
    finally:
        stop["f"] = True
        hc.HF_HUB_OFFLINE = prev_offline

    meta = {"modality": modality, "engine": engine, "display": display or name,
            "role": "model", "source": repo}
    if arch:
        meta["arch"] = arch
    (dest / "mlxstudio.json").write_text(json.dumps(meta, indent=2))

    if on_progress:
        on_progress(1.0)
    return {"id": name, "dir": str(dest), "modality": modality,
            "size_gb": round(_dir_size(dest) / 1e9, 2)}
