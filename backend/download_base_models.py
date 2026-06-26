"""Pull the 3 MLX Creator base models (one per modality), with progress.

  1. FLUX.1-schnell   — image
  2. ACE-Step 1.5     — music   (+ its 5Hz planner)
  3. Wan 2.2 TI2V-5B  — video   (+ its umt5 text tokenizer)

Re-running is a no-op for anything already present (snapshot_download skips
files that exist). No HF login required (all ungated mirrors).
"""
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
import config
MODELS = config.models_dir()

# label -> list of (repo, dest-folder, allow_patterns or None)
BASES = {
    "FLUX.1-schnell (image)": [
        ("lzyvegetable/FLUX.1-schnell", "FLUX.1-schnell",
         ["flux1-schnell.safetensors", "ae.safetensors", "text_encoder/*",
          "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*"]),
    ],
    "ACE-Step 1.5 (music)": [
        ("mlx-community/ACE-Step1.5-MLX", "ACE-Step1.5-MLX", None),
        ("ACE-Step/acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-0.6B", None),
    ],
    "Wan 2.2 TI2V-5B (video)": [
        ("SceneWorks/wan2.2-ti2v-5b-mlx", "Wan2.2-TI2V-5B-MLX", None),
        ("google/umt5-xxl", "umt5-xxl-tokenizer",
         ["tokenizer*", "spiece*", "special_tokens*", "config.json"]),
    ],
}

IGNORE = ["*.md", ".gitattributes", "*.png", "*.jpg", "*.jpeg", "LICENSE", "*.pt", "*.bin"]


def _size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    print("Installing MLX Creator base models (Flux · ACE-Step 1.5 · Wan 2.2)\n")
    total = 0.0
    for i, (label, repos) in enumerate(BASES.items(), 1):
        print(f"→ [{i}/3] Pulling {label} …", flush=True)
        for repo, dest, allow in repos:
            d = MODELS / dest
            d.mkdir(parents=True, exist_ok=True)
            print(f"      • {repo}", flush=True)
            t0 = time.time()
            snapshot_download(repo, local_dir=str(d), allow_patterns=allow,
                              ignore_patterns=IGNORE, max_workers=8)
            sz = _size(d)
            total += sz
            print(f"        ✓ {sz:.1f} GB in {time.time()-t0:.0f}s", flush=True)
        print(f"  ✓ {label} ready\n", flush=True)
    print(f"✓ Pulled the 3 base models ({total:.0f} GB total): "
          f"Flux (image), ACE-Step 1.5 (music), Wan 2.2 (video).")
    print("  More models anytime via the in-app Models tab. Launch: ./run.sh")


if __name__ == "__main__":
    main()
