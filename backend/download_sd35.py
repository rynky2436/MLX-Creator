import os; from pathlib import Path
os.environ["HF_HUB_OFFLINE"]="0"; os.environ.setdefault("HF_XET_HIGH_PERFORMANCE","1")
from huggingface_hub import snapshot_download
M=Path.home()/"mlx-studio"/"models"
jobs=[
  ("argmaxinc/stable-diffusion","SD3-encoders",
     ["clip_l/*","clip_g/*","t5/*","tokenizer_l/*","tokenizer_g/*"]),
  ("argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized","SD3.5-large-4bit",None),
  ("argmaxinc/mlx-stable-diffusion-3.5-large","SD3.5-large",None),
]
for repo,name,allow in jobs:
    dest=M/name; dest.mkdir(parents=True,exist_ok=True)
    print(f"[{name}] downloading {repo} …", flush=True)
    snapshot_download(repo, local_dir=str(dest), allow_patterns=allow,
        ignore_patterns=["*.md",".gitattributes","*.png"], max_workers=8)
    tot=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
    print(f"[{name}] DONE {tot/1e9:.1f}GB", flush=True)
print("ALL SD3.5 DOWNLOADS COMPLETE", flush=True)
