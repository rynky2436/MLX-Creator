"""MLX Creator API — FastAPI server driving the torch-free MLX media engines.

One background worker thread runs generations serially (MLX uses the GPU
exclusively); progress is pushed to all websocket clients in real time.
"""
from __future__ import annotations

import asyncio
import json
import queue
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


def worker() -> None:
    while True:
        job_id = JOB_Q.get()
        job = JOBS[job_id]
        if job.get("status") == "canceled":
            continue
        try:
            _set(job_id, status="running", stage="starting", progress=0.0)

            def on_step(i, total, _jid=job_id):
                _set(_jid, stage="denoising", step=i, total=total,
                     progress=round(i / total, 3))

            def on_stage(s, _jid=job_id):
                _set(_jid, stage=s)

            if job["kind"] == "install":
                def on_prog(p, _jid=job_id):
                    _set(_jid, stage="downloading", progress=round(p, 3))
                result = installer.download_hf(
                    repo=job["repo"], modality=job["modality"], engine=job["engine"],
                    arch=job.get("arch"), display=job.get("display"),
                    on_progress=on_prog, on_stage=on_stage,
                )
            elif job["kind"] == "music":
                result = music_engine.generate(
                    prompt=job["prompt"], lyrics=job.get("lyrics", ""),
                    duration=job["duration"], steps=job["steps"],
                    guidance=job["guidance"], shift=job["shift"],
                    vocal_language=job["vocal_language"], seed=job["seed"],
                    lm_size=job.get("lm_size", "0.6B"),
                    model_id=job.get("model", "ACE-Step1.5-MLX"), on_stage=on_stage,
                )
            elif job["kind"] == "video":
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
            _set(job_id, status="done", stage="done", progress=1.0,
                 result=result, finished_at=time.time())
        except Exception as e:  # surface failures to the UI
            import traceback
            traceback.print_exc()
            _set(job_id, status="error", error=f"{type(e).__name__}: {e}")
        finally:
            JOB_Q.task_done()


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=worker, daemon=True).start()
    asyncio.create_task(_broadcaster())


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
        "lm_size": req.lm_size, "model": req.model, "width": 0, "height": 0,
        "created_at": time.time(),
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


@app.get("/api/browse")
async def api_browse(source: str = "huggingface", modality: str = "image", q: str = ""):
    if source == "civitai":
        return browser.search_civitai_loras(query=q)
    return browser.search_hf(modality, query=q)


@app.get("/api/installed")
async def api_installed(modality: str | None = None):
    return registry.list_installed(modality)


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
    # send a snapshot of current jobs on connect
    for job in sorted(JOBS.values(), key=lambda j: j["created_at"]):
        await ws.send_text(json.dumps({"type": "job", "job": job}))
    try:
        while True:
            await ws.receive_text()  # keepalive / ignore
    except WebSocketDisconnect:
        _clients.discard(ws)


# ---- static: outputs + frontend ----------------------------------------
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))
