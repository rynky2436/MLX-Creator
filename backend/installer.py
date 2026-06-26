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
MODELS = ROOT / "models"

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
