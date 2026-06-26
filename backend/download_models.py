"""One-time downloader: pull FLUX.1-schnell weights into mlx-studio/models/.

Downloads ONLY the files the MLX flux loader needs (BFL single-file layout),
skipping the duplicate diffusers-format transformer/ and vae/ folders.
Re-running is a no-op once files exist. No HF login required (ungated mirror).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

REPO = os.getenv("FLUX_SCHNELL_REPO", "lzyvegetable/FLUX.1-schnell")
DEST = MODELS / "FLUX.1-schnell"

ALLOW = [
    "flux1-schnell.safetensors",   # diffusion model
    "ae.safetensors",              # vae
    "text_encoder/*",              # CLIP
    "text_encoder_2/*",            # T5-XXL (sharded)
    "tokenizer/*",
    "tokenizer_2/*",
]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} -> {DEST}")
    print("Files:", ", ".join(ALLOW))
    snapshot_download(
        repo_id=REPO,
        local_dir=str(DEST),
        allow_patterns=ALLOW,
        max_workers=8,
    )
    # report
    total = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"DONE. {total/1e9:.1f} GB in {DEST}")

if __name__ == "__main__":
    main()
