"""
OMEGA Harvester Service
-----------------------
FastAPI server that receives harvest jobs from the admin panel
and runs them using the appropriate harvester class.

Deploy on Render (free tier) or any container host.
Port: 8000
"""
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import uuid
import yaml
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    AI_ENGINE_API_KEY, MAX_WORKERS,
)
from harvesters import WebHarvester, ImageHarvester, AudioHarvester, VideoHarvester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Supabase client ────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── Active job queue ──────────────────────────────────────────────────────────
_semaphore = asyncio.Semaphore(MAX_WORKERS)

# ── Path to train_textnet.py ───────────────────────────────────────────────────
_HARVESTERS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_SCRIPT = os.path.join(_HARVESTERS_DIR, "train_textnet.py")


def get_harvester(network_type: str):
    return {
        "text":  WebHarvester(supabase),
        "web":   WebHarvester(supabase),
        "image": ImageHarvester(supabase),
        "audio": AudioHarvester(supabase),
        "video": VideoHarvester(supabase),
    }.get(network_type, WebHarvester(supabase))


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from config import USER_AGENT, AI_ENGINE_URL
    logger.info("OMEGA Harvester Service started (USER_AGENT=%s..., AI_ENGINE=%s)", USER_AGENT[:30], AI_ENGINE_URL)
    yield
    logger.info("OMEGA Harvester Service stopped")

app = FastAPI(title="OMEGA Harvester Service", lifespan=lifespan)

# Mount Encoder Service at /encoder (for unified deploy — AI Engine calls ENCODER_SERVICE_URL=http://localhost:8000/encoder)
from encoder_service import app as encoder_app
app.mount("/encoder", encoder_app)


# ── Auth ──────────────────────────────────────────────────────────────────────
def verify_key(x_api_key: str = Header(...)):
    if x_api_key != AI_ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Models ────────────────────────────────────────────────────────────────────
class HarvestRequest(BaseModel):
    source_url:  str
    network_type: str = "text"


class HarvestResponse(BaseModel):
    job_id:  str
    status:  str = "queued"
    message: str


class TrainRequest(BaseModel):
    level:          int | None = None
    hf_dataset:     str | None = None
    hf_config:      str | None = None
    hf_column:      str | None = None
    hf_split:       str | None = None
    streaming:      bool = False
    config:         str | None = None
    url:            str | None = None
    limit:          int | None = None
    lama:           str | None = None
    dataset:        str | None = None
    batch:          int | None = None
    lang:           str | None = None
    sources:        list[dict] | None = None
    duration_hours: float | None = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name":    "OMEGA Harvester Service",
        "version": "1.0",
        "status":  "running",
        "docs":    {
            "health": "GET /health",
            "harvest": "POST /harvest (x-api-key required)",
            "train":   "POST /train (x-api-key required)",
            "job":     "GET /jobs/{job_id}",
            "swagger": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    return {"ok": True, "workers": MAX_WORKERS}


@app.post("/harvest", response_model=HarvestResponse)
async def start_harvest(req: HarvestRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    job_id = str(uuid.uuid4())
    supabase.table("harvester_jobs").insert({
        "id":          job_id,
        "type":        req.network_type,
        "source_url":  req.source_url,
        "status":      "queued",
        "started_at":  datetime.utcnow().isoformat(),
    }).execute()

    asyncio.create_task(_run_job(job_id, req.source_url, req.network_type))

    return HarvestResponse(
        job_id=job_id,
        status="queued",
        message=f"Harvest job {job_id} queued",
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    res = supabase.table("harvester_jobs").select("*").eq("id", job_id).single().execute()
    if not res.data:
        raise HTTPException(404, "Job not found")
    return res.data


def _write_temp_config(sources: list[dict]) -> str:
    """Write sources to a temp YAML file. Returns path. Caller must delete."""
    config = {"sources": sources}
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="omega_train_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return path
    except Exception:
        os.unlink(path)
        raise


@app.post("/train")
async def start_train(req: TrainRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    job_id = str(uuid.uuid4())
    args = [sys.executable, _TRAIN_SCRIPT]
    temp_config_path: str | None = None

    # If sources provided, write temp YAML and use --config
    if req.sources and len(req.sources) > 0:
        temp_config_path = _write_temp_config(req.sources)
        args.extend(["--config", temp_config_path])
    elif req.config:
        args.extend(["--config", req.config])

    if req.duration_hours is not None and req.duration_hours > 0:
        args.extend(["--duration-hours", str(req.duration_hours)])

    if req.level is not None:
        args.extend(["--level", str(req.level)])
    if req.hf_dataset and not req.sources:
        args.extend(["--hf-dataset", req.hf_dataset])
    if req.hf_config and not req.sources:
        args.extend(["--hf-config", req.hf_config])
    if req.hf_column and not req.sources:
        args.extend(["--hf-column", req.hf_column])
    if req.hf_split and not req.sources:
        args.extend(["--hf-split", req.hf_split])
    if req.streaming:
        args.append("--streaming")
    if req.url:
        args.extend(["--url", req.url])
    if req.limit is not None:
        args.extend(["--limit", str(req.limit)])
    if req.lama:
        args.extend(["--lama", req.lama])
    if req.dataset:
        args.extend(["--dataset", req.dataset])
    if req.batch is not None:
        args.extend(["--batch", str(req.batch)])
    if req.lang:
        args.extend(["--lang", req.lang])

    # Determine source for harvest_log
    if req.sources and len(req.sources) > 0:
        source_type, source_id = "sources", f"{len(req.sources)}_sources"
    elif req.hf_dataset:
        source_type, source_id = "huggingface", req.hf_dataset
    elif req.config:
        source_type, source_id = "config", req.config
    elif req.url:
        source_type, source_id = "url", req.url
    elif req.lama:
        source_type, source_id = "lama", req.lama
    elif req.dataset:
        source_type, source_id = "jsonl", req.dataset
    else:
        source_type, source_id = "wikipedia", f"level_{req.level or 1}"

    config = {}
    if req.limit is not None:
        config["limit"] = req.limit
    if req.hf_column:
        config["column"] = req.hf_column
    if req.streaming:
        config["streaming"] = True
    if req.duration_hours is not None:
        config["duration_hours"] = req.duration_hours

    try:
        supabase.table("harvest_log").upsert({
            "job_id": job_id,
            "source_type": source_type,
            "source_id": source_id,
            "config": config or None,
            "status": "running",
        }, on_conflict="job_id").execute()
    except Exception as e:
        logger.warning("harvest_log start failed: %s", e)

    # Логируем stderr в файл — при ошибке HF/train можно посмотреть
    log_dir = os.path.join(_HARVESTERS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{job_id[:8]}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            args,
            cwd=_HARVESTERS_DIR,
            env={**os.environ},
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    logger.info("Train job %s started, log: %s", job_id[:8], log_path)

    return {"job_id": job_id, "status": "started", "message": f"Training job {job_id} started", "log_file": f"logs/train_{job_id[:8]}.log"}


# ── Background task ───────────────────────────────────────────────────────────
async def _run_job(job_id: str, source: str, network_type: str):
    async with _semaphore:
        harvester = get_harvester(network_type)
        try:
            count = await harvester.run(source, job_id)
            logger.info("Job %s completed — %d items harvested from %s", job_id, count, source)
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e)
        finally:
            await harvester.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
