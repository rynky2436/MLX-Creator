"""MLX Creator API — FastAPI server driving the torch-free MLX media engines.

One background worker thread runs generations serially (MLX uses the GPU
exclusively); progress is pushed to all websocket clients in real time.
"""
from __future__ import annotations

import asyncio
import json
import queue
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import flux_engine
import sd35_engine
import qwen_engine
import music_engine
import video_engine
import browser
import registry
import installer
import config
import shutil
import mlx.core as mx

# Keep exactly ONE model resident: switching engine/model frees the previous
# one and clears MLX's buffer cache; the same model stays loaded between gens.
_LOADED = {"key": None}


def ensure_only_loaded(engine: str, model: str) -> None:
    key = (engine, model)
    if _LOADED["key"] == key:
        return
    # Drop the previous model's arrays, then return MLX's freed buffer pool to
    # the OS so RSS actually shrinks on a model switch.
    for eng in (flux_engine, sd35_engine, music_engine):
        try:
            eng.unload()
        except Exception:
            pass
    try:
        (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
    except Exception:
        pass
    _LOADED["key"] = key


def free_memory() -> None:
    """Unload every resident model and return MLX's buffer pool to the OS.
    The next generation reloads its model on demand."""
    for eng in (flux_engine, sd35_engine, qwen_engine, music_engine, video_engine):
        try:
            eng.unload()
        except Exception:
            pass
    try:
        (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
    except Exception:
        pass
    import gc
    gc.collect()
    had = _LOADED["key"]
    _LOADED["key"] = None
    print(f"[mem] released resident model ({had or 'none'})", flush=True)


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FRONTEND = ROOT / "frontend"

app = FastAPI(title="MLX Creator")

# ---- job state ----------------------------------------------------------
JOBS: dict[str, dict] = {}
JOB_Q: "queue.Queue[str]" = queue.Queue()
_loop: asyncio.AbstractEventLoop | None = None
_clients: set[WebSocket] = set()
_event_q: "asyncio.Queue[dict]" = asyncio.Queue()
_idle_task: "asyncio.Task | None" = None
IDLE_FREE_SECS = 30   # free the model this long after the last UI window closes


async def _idle_free():
    try:
        await asyncio.sleep(IDLE_FREE_SECS)
    except asyncio.CancelledError:
        return
    if _clients:
        return  # a window reopened in the meantime
    if any(j.get("status") in ("running", "queued") for j in JOBS.values()):
        return  # work in progress — keep the model loaded
    free_memory()


def _on_clients_changed():
    """Cancel any pending free while a window is open; schedule one once the
    last window closes, so closing the UI actually releases the model's RAM."""
    global _idle_task
    if _idle_task and not _idle_task.done():
        _idle_task.cancel()
        _idle_task = None
    if not _clients:
        _idle_task = asyncio.create_task(_idle_free())


class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "FLUX.1-schnell"
    negative_prompt: str = ""
    steps: int | None = None
    guidance: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    quantize: bool = True


class MusicRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    lyrics: str = ""
    duration: float = 30.0
    steps: int = 20
    guidance: float = 1.0
    shift: float = 3.0
    vocal_language: str = "unknown"
    seed: int | None = None
    lm_size: str = "0.6B"
    model: str = "ACE-Step1.5-MLX"
    title: str = ""


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    model: str = "Wan2.2-TI2V-5B-MLX"
    width: int = 704
    height: int = 480
    num_frames: int = 49
    duration: float | None = None
    steps: int = 20
    guidance: float | None = None
    seed: int = -1


class InstallRequest(BaseModel):
    repo: str = Field(..., min_length=1)
    modality: str
    engine: str
    arch: str | None = None
    display: str | None = None


def emit(event: dict) -> None:
    """Thread-safe: push an event onto the asyncio queue from the worker."""
    if _loop is not None:
        _loop.call_soon_threadsafe(_event_q.put_nowait, event)


def _set(job_id: str, **kw) -> None:
    JOBS[job_id].update(kw)
    emit({"type": "job", "job": JOBS[job_id]})


CANCEL: set[str] = set()   # job ids asked to stop mid-run


class _Canceled(Exception):
    """Raised from on_step/on_stage/should_cancel to abort a running job."""


def worker() -> None:
    while True:
        job_id = JOB_Q.get()
        job = JOBS[job_id]
        if job.get("status") == "canceled" or job_id in CANCEL:
            CANCEL.discard(job_id)
            _set(job_id, status="canceled", stage="canceled")
            JOB_Q.task_done()
            continue
        try:
            _set(job_id, status="running", stage="starting", progress=0.0)

            def on_step(i, total, _jid=job_id):
                if _jid in CANCEL:
                    raise _Canceled()
                _set(_jid, stage="denoising", step=i, total=total,
                     progress=round(i / total, 3))

            def on_stage(s, _jid=job_id):
                if _jid in CANCEL:
                    raise _Canceled()
                _set(_jid, stage=s)

            def should_cancel(_jid=job_id):
                if _jid in CANCEL:
                    raise _Canceled()

            if job["kind"] == "install":
                def on_prog(p, _jid=job_id):
                    _set(_jid, stage="downloading", progress=round(p, 3))
                result = installer.download_hf(
                    repo=job["repo"], modality=job["modality"], engine=job["engine"],
                    arch=job.get("arch"), display=job.get("display"),
                    on_progress=on_prog, on_stage=on_stage,
                )
            elif job["kind"] == "install_base":
                def on_prog(p, _jid=job_id):
                    _set(_jid, stage="downloading", progress=round(p, 3))
                result = installer.download_base(
                    job["base"], on_progress=on_prog, on_stage=on_stage)
            elif job["kind"] == "install_planner":
                def on_prog(p, _jid=job_id):
                    _set(_jid, stage="downloading", progress=round(p, 3))
                result = installer.download_planner4b(
                    on_progress=on_prog, on_stage=on_stage)
            elif job["kind"] == "music":
                ensure_only_loaded("ace_step", job.get("model", "ACE-Step1.5-MLX"))
                result = music_engine.generate(
                    prompt=job["prompt"], lyrics=job.get("lyrics", ""),
                    duration=job["duration"], steps=job["steps"],
                    guidance=job["guidance"], shift=job["shift"],
                    vocal_language=job["vocal_language"], seed=job["seed"],
                    lm_size=job.get("lm_size", "0.6B"),
                    model_id=job.get("model", "ACE-Step1.5-MLX"), on_stage=on_stage,
                    should_cancel=should_cancel, title=job.get("title", ""),
                )
            elif job["kind"] == "video":
                ensure_only_loaded("wan", job.get("model", "Wan2.2-TI2V-5B-MLX"))
                result = video_engine.generate(
                    prompt=job["prompt"], negative_prompt=job.get("negative_prompt"),
                    width=job["width"], height=job["height"],
                    num_frames=job["num_frames"], duration=job.get("duration"),
                    steps=job["steps"], guidance=job["guidance"], seed=job["seed"],
                    model_id=job.get("model", "Wan2.2-TI2V-5B-MLX"), on_stage=on_stage,
                )
            else:  # image — route by the selected model's engine
                eng = next((m.get("engine") for m in registry.list_installed("image")
                            if m["id"] == job["model"]), "flux")
                ensure_only_loaded(eng, job["model"])
                if eng == "sd35":
                    result = sd35_engine.generate(
                        prompt=job["prompt"], model=job["model"], steps=job["steps"],
                        guidance=job["guidance"], width=job["width"], height=job["height"],
                        seed=job["seed"], negative_prompt=job.get("negative_prompt", ""),
                        on_stage=on_stage,
                    )
                elif eng == "qwen":
                    result = qwen_engine.generate(
                        prompt=job["prompt"], model=job["model"], steps=job["steps"],
                        guidance=job["guidance"], width=job["width"], height=job["height"],
                        seed=job["seed"], negative_prompt=job.get("negative_prompt", ""),
                        on_stage=on_stage,
                    )
                else:
                    result = flux_engine.generate(
                        prompt=job["prompt"], model=job["model"], steps=job["steps"],
                        guidance=job["guidance"], width=job["width"], height=job["height"],
                        seed=job["seed"], quantize=job["quantize"],
                        on_step=on_step, on_stage=on_stage,
                    )
            if job_id in CANCEL:
                raise _Canceled()
            _set(job_id, status="done", stage="done", progress=1.0,
                 result=result, finished_at=time.time())
        except _Canceled:
            _set(job_id, status="canceled", stage="canceled", finished_at=time.time())
        except Exception as e:  # surface failures to the UI
            import traceback
            traceback.print_exc()
            if job_id in CANCEL:
                _set(job_id, status="canceled", stage="canceled")
            else:
                _set(job_id, status="error", error=f"{type(e).__name__}: {e}")
        finally:
            CANCEL.discard(job_id)
            JOB_Q.task_done()


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=worker, daemon=True).start()
    asyncio.create_task(_broadcaster())


@app.on_event("shutdown")
async def _shutdown():
    # Free the resident model on any clean shutdown (Ctrl+C, app quit, SIGTERM).
    free_memory()


async def _broadcaster():
    while True:
        event = await _event_q.get()
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)
        if dead:
            _on_clients_changed()  # a window dropped → maybe schedule model free


# ---- API ----------------------------------------------------------------
@app.post("/api/generate")
async def api_generate(req: GenRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "image", "status": "queued", "stage": "queued",
        "progress": 0.0, "prompt": req.prompt, "model": req.model, "steps": req.steps,
        "negative_prompt": req.negative_prompt,
        "guidance": req.guidance, "width": req.width, "height": req.height,
        "seed": req.seed, "quantize": req.quantize, "created_at": time.time(),
    }
    JOB_Q.put(job_id)
    emit({"type": "job", "job": JOBS[job_id]})
    return {"job_id": job_id, "queue_pos": JOB_Q.qsize()}


@app.post("/api/generate_music")
async def api_generate_music(req: MusicRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "music", "status": "queued", "stage": "queued",
        "progress": 0.0, "prompt": req.prompt, "lyrics": req.lyrics,
        "duration": req.duration, "steps": req.steps, "guidance": req.guidance,
        "shift": req.shift, "vocal_language": req.vocal_language, "seed": req.seed,
        "lm_size": req.lm_size, "model": req.model, "title": req.title,
        "width": 0, "height": 0, "created_at": time.time(),
    }
    JOB_Q.put(job_id)
    emit({"type": "job", "job": JOBS[job_id]})
    return {"job_id": job_id, "queue_pos": JOB_Q.qsize()}


@app.post("/api/generate_video")
async def api_generate_video(req: VideoRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "video", "status": "queued", "stage": "queued",
        "progress": 0.0, "prompt": req.prompt, "negative_prompt": req.negative_prompt,
        "width": req.width, "height": req.height, "num_frames": req.num_frames,
        "duration": req.duration, "steps": req.steps, "guidance": req.guidance,
        "seed": req.seed, "model": req.model, "created_at": time.time(),
    }
    JOB_Q.put(job_id)
    emit({"type": "job", "job": JOBS[job_id]})
    return {"job_id": job_id, "queue_pos": JOB_Q.qsize()}


@app.get("/api/jobs")
async def api_jobs():
    return JSONResponse(sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True))


@app.post("/api/cancel")
async def api_cancel(req: dict):
    jid = req.get("id")
    j = JOBS.get(jid)
    if not j:
        return {"ok": False}
    if j.get("status") in ("queued",):
        j["status"] = "canceled"
        j["stage"] = "canceled"
        emit({"type": "job", "job": j})
    elif j.get("status") == "running":
        CANCEL.add(jid)
        _set(jid, stage="canceling")
    return {"ok": True}


@app.post("/api/jobs/delete")
async def api_jobs_delete(req: dict):
    jid = req.get("id")
    CANCEL.discard(jid)
    if JOBS.pop(jid, None):
        emit({"type": "job_removed", "id": jid})
    return {"ok": True}


@app.get("/api/browse")
async def api_browse(source: str = "huggingface", modality: str = "image", q: str = ""):
    return browser.search_hf(modality, query=q)


@app.get("/api/installed")
async def api_installed(modality: str | None = None):
    return registry.list_installed(modality)


@app.get("/api/base_models")
async def api_base_models():
    return installer.base_models_status()


def _dir_gb(d: Path) -> float:
    return round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9, 2)


@app.get("/api/settings")
async def api_settings():
    md = config.models_dir()
    try:
        free = shutil.disk_usage(md).free / 1e9
    except Exception:
        free = 0
    models = registry.list_installed()
    for m in models:
        m["size_gb"] = _dir_gb(Path(m["dir"]))
    return {"models_dir": str(md), "models_total_gb": _dir_gb(md),
            "disk_free_gb": round(free, 1), "models": models}


@app.post("/api/settings")
async def api_set_settings(req: dict):
    out = {"ok": True}
    if "models_dir" in req:
        p = Path(req["models_dir"]).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("x")
            t.unlink()
        except Exception as e:
            return JSONResponse({"error": f"Can't use that folder: {e}"}, status_code=400)
        config.update({"models_dir": str(p)})
        out["restart_required"] = True
    out["settings"] = config.load()
    return out


@app.post("/api/uninstall")
async def api_uninstall(req: dict):
    md = config.models_dir().resolve()
    target = (md / req.get("id", "")).resolve()
    if md not in target.parents or not target.is_dir():
        return JSONResponse({"error": "invalid model id"}, status_code=400)
    freed = _dir_gb(target)
    shutil.rmtree(target)
    return {"ok": True, "freed_gb": freed}


@app.post("/api/free")
async def api_free():
    free_memory()
    return {"ok": True}


@app.post("/api/shutdown")
async def api_shutdown():
    """Stop the server and release all its memory back to the OS."""
    free_memory()

    def _die():
        time.sleep(0.3)  # let the HTTP response flush first
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"ok": True}


@app.post("/api/install_base")
async def api_install_base(req: dict):
    ids = []
    for base in req.get("bases", []):
        if base not in installer.BASE_RECIPES:
            continue
        rec = installer.BASE_RECIPES[base]
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {
            "id": job_id, "kind": "install_base", "status": "queued",
            "stage": "queued", "progress": 0.0, "prompt": rec["display"],
            "base": base, "modality": rec["modality"], "model": "installer",
            "created_at": time.time(),
        }
        JOB_Q.put(job_id)
        emit({"type": "job", "job": JOBS[job_id]})
        ids.append(job_id)
    return {"job_ids": ids}


@app.post("/api/install_planner")
async def api_install_planner():
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "install_planner", "status": "queued",
        "stage": "queued", "progress": 0.0, "prompt": "ACE-Step Planner 4B",
        "modality": "audio", "model": "installer", "created_at": time.time(),
    }
    JOB_Q.put(job_id)
    emit({"type": "job", "job": JOBS[job_id]})
    return {"job_id": job_id}


@app.post("/api/install")
async def api_install(req: InstallRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "install", "status": "queued", "stage": "queued",
        "progress": 0.0, "prompt": req.display or req.repo, "repo": req.repo,
        "modality": req.modality, "engine": req.engine, "arch": req.arch,
        "display": req.display, "model": "installer", "created_at": time.time(),
    }
    JOB_Q.put(job_id)
    emit({"type": "job", "job": JOBS[job_id]})
    return {"job_id": job_id}


@app.get("/api/models")
async def api_models():
    return {**flux_engine.model_status(), **music_engine.model_status(),
            **video_engine.model_status()}


_TYPE = {".png": "image", ".wav": "audio", ".mp4": "video"}


@app.get("/api/gallery")
async def api_gallery():
    files = [p for p in OUTPUTS.iterdir() if p.suffix.lower() in _TYPE]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"filename": p.name, "url": f"/outputs/{p.name}",
             "type": _TYPE[p.suffix.lower()],
             "mtime": p.stat().st_mtime} for p in files]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    _on_clients_changed()  # a window is open — cancel any pending idle free
    # send a snapshot of current jobs on connect
    for job in sorted(JOBS.values(), key=lambda j: j["created_at"]):
        await ws.send_text(json.dumps({"type": "job", "job": job}))
    try:
        while True:
            await ws.receive_text()  # keepalive / ignore
    except WebSocketDisconnect:
        _clients.discard(ws)
        _on_clients_changed()  # last window closed → schedule model free


# ---- static: outputs + frontend ----------------------------------------
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))
