"""Model browser — search HuggingFace (MLX models) and Civitai (Flux LoRAs),
normalize results, and detect which are drop-in compatible with our engines.

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
        # mlx-audio loads many model_types from config.json
        return {"compat": "drop-in", "engine": "mlx-audio", "note": "mlx-audio model"}

    if engine == "mlx-video":
        is_wan = "wan" in name
        is_ltx = "ltx" in name
        if is_wan or is_ltx:
            return {"compat": "drop-in", "engine": "mlx-video",
                    "note": "Wan" if is_wan else "LTX"}
        return {"compat": "maybe", "engine": "mlx-video", "note": "verify on install"}

    return {"compat": "unknown", "engine": engine, "note": ""}


def search_hf(modality: str, query: str = "", limit: int = 30) -> list[dict]:
    spec = MODALITIES.get(modality)
    if not spec:
        return []
    out, seen = [], set()
    for ptag in spec["tags"]:
        params = {"filter": "mlx", "pipeline_tag": ptag, "sort": "downloads",
                  "limit": limit, "full": "true"}
        if query:
            params["search"] = query
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
    out.sort(key=lambda x: x["downloads"], reverse=True)
    return out[:limit]


# ---- Civitai (Flux LoRAs only) ----------------------------------------
def search_civitai_loras(query: str = "", limit: int = 30) -> list[dict]:
    params = {"types": "LORA", "limit": limit, "sort": "Highest Rated",
              "baseModels": "Flux.1 D"}  # server-side filter to Flux LoRAs
    if query:
        params["query"] = query
    try:
        data = _get("https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params))
    except Exception:
        return []
    out = []
    for m in data.get("items", []):
        vers = m.get("modelVersions") or []
        if not vers:
            continue
        v = vers[0]
        base = (v.get("baseModel") or "")
        if "flux" not in base.lower():
            continue  # only Flux LoRAs are usable by our engine
        files = v.get("files") or []
        f0 = next((f for f in files if f.get("downloadUrl")), files[0] if files else {})
        imgs = v.get("images") or []
        out.append({
            "source": "civitai", "id": str(m["id"]), "name": m["name"],
            "author": (m.get("creator") or {}).get("username", "?"),
            "modality": "lora", "base_model": base,
            "downloads": (m.get("stats") or {}).get("downloadCount", 0),
            "size_gb": round((f0.get("sizeKB") or 0) / 1e6, 3),
            "version_id": v.get("id"),
            "download_url": f0.get("downloadUrl"),
            "filename": f0.get("name"),
            "thumb": imgs[0]["url"] if imgs and imgs[0].get("url") else None,
            "url": f"https://civitai.com/models/{m['id']}",
            "compat": "lora", "engine": "flux",
            "note": base,
        })
    return out
