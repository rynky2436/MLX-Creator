"""Model browser — search HuggingFace, normalize results, and detect which are
drop-in compatible with our engines.

No torch, no downloads here — just metadata queries used by the Models tab.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

UA = {"User-Agent": "mlx-studio"}

# modality -> (HF pipeline_tags, target engine in our app)
MODALITIES = {
    "image": {"tags": ["text-to-image"], "engine": "flux"},
    "video": {"tags": ["text-to-video", "image-to-video"], "engine": "mlx-video"},
    "audio": {"tags": ["text-to-audio", "text-to-speech"], "engine": "mlx-audio"},
}


def _get(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---- HuggingFace -------------------------------------------------------
# Runtimes that require torch — unusable under the MLX-only rule.
_TORCH_RUNTIMES = {"mflux", "diffusionkit", "diffusers", "transformers", "torch"}


def _classify_hf(model_id: str, tags: list[str], siblings: list[str],
                 engine: str, library: str | None = None) -> dict:
    """Compatibility for OUR torch-free loaders. The library_name is the real
    tell: mflux/diffusionkit/diffusers = torch (can't run here)."""
    files = set(siblings)
    name = model_id.lower()
    lib = (library or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    is_mlx = lib == "mlx" or "mlx" in tagset or "mlx" in name

    # Universal format gates — these never load in our MLX engines, regardless
    # of modality. Catch them before the optimistic per-engine branches.
    if any(f.endswith(".gguf") for f in files) or "gguf" in name:
        return {"compat": "unsupported", "engine": None, "note": "GGUF (llama.cpp) — not MLX"}
    if "comfyui" in name or "comfy-org" in name or "comfy_org" in name or "comfyui" in tagset:
        return {"compat": "unsupported", "engine": None, "note": "ComfyUI package — not a loadable model"}
    if not is_mlx and any(k in name for k in ("fp8", "nvfp4", "mxfp4", "df11", "awq", "gptq", "-nf4")):
        return {"compat": "needs-torch", "engine": None, "note": "torch quant format — not MLX"}

    # We vendor DiffusionKit's MLX SD3/SD3.5 path (torch-free) and mflux's Qwen.
    if lib == "diffusionkit" and ("stable-diffusion-3" in name or "sd3" in name):
        return {"compat": "drop-in", "engine": "sd35",
                "note": "SD3/3.5 via DiffusionKit (pulls encoder bundle)"}
    if lib == "mflux" and "qwen-image" in name:
        return {"compat": "drop-in", "engine": "qwen", "note": "Qwen-Image via mflux"}

    # Hard gate: other torch runtimes → not loadable in this MLX-only app.
    if lib in _TORCH_RUNTIMES:
        return {"compat": "needs-torch", "engine": None,
                "note": f"{lib} runtime — needs torch"}

    if engine == "flux":
        # our flux engine wants the mlx-examples / BFL single-file layout
        bfl = any("flux1-" in f and f.endswith(".safetensors") for f in files) \
            and any(f == "ae.safetensors" for f in files)
        is_flux = "flux" in name
        if bfl:
            return {"compat": "drop-in", "engine": "flux", "note": "BFL layout"}
        if is_flux and ("mflux" in name or "4bit" in name or "diffusionkit" in name):
            return {"compat": "convert", "engine": "flux", "note": "mflux/DiffusionKit format — needs remap"}
        if is_flux:
            return {"compat": "maybe", "engine": "flux", "note": "Flux — verify layout on install"}
        return {"compat": "unsupported", "engine": None, "note": "non-Flux image model"}

    if engine == "mlx-audio":
        # MLX-converted audio = drop-in; otherwise it may need conversion, so
        # don't pretend it's a guaranteed drop-in.
        if is_mlx:
            return {"compat": "drop-in", "engine": "mlx-audio", "note": "MLX audio model"}
        return {"compat": "maybe", "engine": "mlx-audio", "note": "verify MLX layout on install"}

    if engine == "mlx-video":
        is_wan = "wan" in name
        is_ltx = "ltx" in name
        if is_wan or is_ltx:
            return {"compat": "drop-in", "engine": "mlx-video",
                    "note": "Wan" if is_wan else "LTX"}
        return {"compat": "maybe", "engine": "mlx-video", "note": "verify on install"}

    return {"compat": "unknown", "engine": engine, "note": ""}


# Our own published models (MLXCreator/*) — surfaced at the top of the browser
# as official drop-ins. (repo_id, engine, arch). Brand-new repos have ~0
# downloads + no pipeline_tag, so they'd never rank in the generic search.
FEATURED = {
    "image": [
        ("MLXCreator/MLXCreator-Flux-Schnell", "flux", "schnell"),
        ("MLXCreator/MLXCreator-SD3.5-Large", "sd35", None),
        ("MLXCreator/MLXCreator-SD3.5-Large-4bit", "sd35", None),
        ("MLXCreator/MLXCreator-SD3-Medium", "sd35", None),
        ("MLXCreator/MLXCreator-QwenImage-4bit", "qwen", None),
        ("MLXCreator/MLXCreator-QwenImage-8bit", "qwen", None),
    ],
    "audio": [
        ("MLXCreator/MLXCreator-ACEStep-1.5", "ace_step", None),
        ("MLXCreator/MLXCreator-ACEStep-1.5-4bit", "ace_step", None),
    ],
    "video": [
        ("MLXCreator/MLXCreator-Wan2.2-TI2V-5B", "wan", None),
    ],
}


def _featured(modality: str, query: str = "") -> list[dict]:
    items = FEATURED.get(modality, [])
    if not items:
        return []
    try:
        res = _get("https://huggingface.co/api/models?author=MLXCreator&full=true&limit=100")
        info = {m["id"]: m for m in res}
    except Exception:
        info = {}
    q = query.lower()
    out = []
    for rid, engine, arch in items:
        if q and q not in rid.lower():
            continue
        m = info.get(rid, {})
        sibs = [s.get("rfilename", "") for s in (m.get("siblings") or [])]
        size_b = sum(s.get("size") or 0 for s in (m.get("siblings") or []))
        out.append({
            "source": "huggingface", "id": rid, "name": rid.split("/")[-1],
            "author": "MLXCreator", "modality": modality,
            "downloads": m.get("downloads", 0), "likes": m.get("likes", 0),
            "size_gb": round(size_b / 1e9, 2) if size_b else None,
            "url": f"https://huggingface.co/{rid}", "files": len(sibs),
            "official": True, "compat": "drop-in", "engine": engine, "arch": arch,
            "note": "MLX Creator — official",
        })
    return out


def search_hf(modality: str, query: str = "", limit: int = 30) -> list[dict]:
    spec = MODALITIES.get(modality)
    if not spec:
        return []
    q = query.strip()
    raw = max(limit, 40)

    # A typed query does a broad name search across ALL of HF — NOT gated on the
    # `mlx` tag or a pipeline_tag, since plenty of real repos (ACE-Step variants,
    # community ports) lack those. Compatibility is surfaced per-result via the
    # badge instead of hiding the model. Browsing (no query) keeps the tighter
    # mlx + task filter for relevance.
    param_sets = []
    if q:
        param_sets.append({"search": q, "sort": "downloads", "limit": raw, "full": "true"})
        param_sets.append({"search": q, "filter": "mlx", "limit": raw, "full": "true"})
    else:
        for ptag in spec["tags"]:
            param_sets.append({"filter": "mlx", "pipeline_tag": ptag,
                               "sort": "downloads", "limit": raw, "full": "true"})

    out, seen = [], set()
    for params in param_sets:
        try:
            results = _get("https://huggingface.co/api/models?" + urllib.parse.urlencode(params))
        except Exception:
            continue
        for m in results:
            mid = m["id"]
            if mid in seen:
                continue
            seen.add(mid)
            sibs = [s.get("rfilename", "") for s in (m.get("siblings") or [])]
            size_b = sum(s.get("size") or 0 for s in (m.get("siblings") or []))
            cls = _classify_hf(mid, m.get("tags", []), sibs, spec["engine"],
                               m.get("library_name"))
            out.append({
                "source": "huggingface", "id": mid, "name": mid.split("/")[-1],
                "author": mid.split("/")[0], "modality": modality,
                "downloads": m.get("downloads", 0), "likes": m.get("likes", 0),
                "size_gb": round(size_b / 1e9, 2) if size_b else None,
                "url": f"https://huggingface.co/{mid}", "files": len(sibs),
                **cls,
            })
    # usable models first (drop-in/convert/maybe), then by downloads — but the
    # incompatible ones still show so you can see the model exists.
    out.sort(key=lambda x: (x.get("compat") in ("needs-torch", "unsupported"),
                            -(x.get("downloads") or 0)))
    feat = _featured(modality, query)
    fids = {f["id"] for f in feat}
    out = [o for o in out if o["id"] not in fids]
    return (feat + out)[:limit]
