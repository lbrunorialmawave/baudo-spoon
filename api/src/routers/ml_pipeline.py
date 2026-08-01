"""Admin router: trigger and monitor the ML training pipeline.

Endpoints
---------
POST /admin/ml/train         — Launch `python -m ml.run_pipeline` in the background
GET  /admin/ml/train/status  — Current status (idle/running/completed/failed)

The pipeline (ml/run_pipeline.py -> ml/pipeline/trainer.py) does role-partitioned
model training, walk-forward backtesting and clustering — realistically minutes,
not seconds, so unlike POST /mantra/run it must not block the request. It is
launched as a detached subprocess (same `python -m ml.run_pipeline` CLI entry
point already used in Docker/manually — see DOCKER_GUIDE.txt), and a background
task waits for it to finish and updates a status file on disk. Completed runs
already show up via the existing GET /model-metrics/runs (populated by the
pipeline itself); this router only adds the "start it and watch progress" half.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import ORJSONResponse

from ..config import settings
from ..deps import require_admin

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/ml",
    tags=["admin-ml"],
    dependencies=[Depends(require_admin)],
)

# A run stuck in "running" for longer than this is treated as stale (e.g. the
# API process restarted mid-training, losing the watcher task) so a new run
# isn't permanently blocked.
_STALE_AFTER_SECONDS = 2 * 60 * 60

# Repo root: api/src/routers/ml_pipeline.py -> api/src/routers -> api/src -> api -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _artifacts_dir() -> Path:
    return Path(settings.artifacts_dir) if hasattr(settings, "artifacts_dir") else Path("artifacts")


def _status_path() -> Path:
    return _artifacts_dir() / "training_status.json"


def _log_path() -> Path:
    return _artifacts_dir() / "training.log"


def _read_status() -> dict:
    path = _status_path()
    if not path.exists():
        return {"status": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "idle"}


def _write_status(data: dict) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_log_tail(max_chars: int = 4000) -> str:
    path = _log_path()
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _is_stale(status: dict) -> bool:
    started_at = status.get("started_at")
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return True
    age = (datetime.now(timezone.utc) - started).total_seconds()
    return age > _STALE_AFTER_SECONDS


async def _watch_training(proc: subprocess.Popen) -> None:
    """Background task: wait for the subprocess, then record the outcome."""
    loop = asyncio.get_event_loop()
    returncode = await loop.run_in_executor(None, proc.wait)

    prior = _read_status()
    _write_status({
        **prior,
        "status": "completed" if returncode == 0 else "failed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "log_tail": _read_log_tail(),
    })
    log.info("ML pipeline finished with returncode=%s", returncode)


@router.post("/train", summary="Launch the ML training pipeline (admin required)")
async def trigger_training(background_tasks: BackgroundTasks) -> ORJSONResponse:
    """Start `python -m ml.run_pipeline` as a background subprocess.

    Returns immediately; poll GET /admin/ml/train/status for progress.
    Rejects a second concurrent launch unless the previous "running" status
    is stale (older than _STALE_AFTER_SECONDS — likely an orphaned run from
    a server restart).
    """
    current = _read_status()
    if current.get("status") == "running" and not _is_stale(current):
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress. Check GET /admin/ml/train/status.",
        )

    sync_url = (
        settings.database_url
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgres+asyncpg://", "postgres+psycopg2://")
    )
    env = {**os.environ, "ML_DATABASE_URL": sync_url}

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            ["python", "-m", "ml.run_pipeline", "--predict-next", "--evaluate-mantra"],
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log.exception("Failed to launch ML training pipeline")
        raise HTTPException(status_code=500, detail="Failed to launch training pipeline. Check server logs.")
    finally:
        # The child inherits its own duplicated file descriptor at spawn time —
        # safe (and necessary, to avoid leaking the fd) to close our copy here.
        log_fh.close()

    started_at = datetime.now(timezone.utc).isoformat()
    _write_status({"status": "running", "started_at": started_at, "pid": proc.pid})

    background_tasks.add_task(_watch_training, proc)

    return ORJSONResponse({"status": "running", "started_at": started_at, "pid": proc.pid})


@router.get("/train/status", summary="Current ML training status")
async def get_training_status() -> ORJSONResponse:
    status = _read_status()
    if status.get("status") == "running" and _is_stale(status):
        status = {**status, "status": "stale"}
    return ORJSONResponse(status)
