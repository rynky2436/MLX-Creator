import os
from pathlib import Path
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE","1")
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/"models"/"ACE-Step1.5-MLX"
DEST.mkdir(parents=True, exist_ok=True)
REPO=os.getenv("ACESTEP_REPO","mlx-community/ACE-Step1.5-MLX")
print(f"Downloading {REPO} -> {DEST}")
snapshot_download(repo_id=REPO, local_dir=str(DEST),
                  ignore_patterns=["*.md",".gitattributes"], max_workers=8)
tot=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
print(f"DONE {tot/1e9:.1f} GB")
