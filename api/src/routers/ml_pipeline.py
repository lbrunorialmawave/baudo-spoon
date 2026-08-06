"""Admin router: trigger and monitor ML training via the GitHub Actions workflow.

Endpoints
---------
POST /admin/ml/train         — Dispatch the "ML Training" GitHub Actions workflow
GET  /admin/ml/train/status  — Status of the most recent run of that workflow

The ml pipeline (ml/run_pipeline.py -> ml/pipeline/trainer.py) needs
dependencies (xgboost, shap, matplotlib, ...) that the API's own Docker image
deliberately does not install — api/requirements.txt and ml/requirements.txt
are disjoint on purpose, since api and ml are separate deployable images.
Running the pipeline as a subprocess inside the API process therefore cannot
work in production (the API container has neither the ml/ source at the
expected path nor its dependencies installed).

The pipeline already has a working, production-secrets-configured entry
point: .github/workflows/ml-training.yml (workflow_dispatch), which builds
ml/Dockerfile fresh on GitHub's own runner and writes results to the same
DB/R2 storage the API reads from. This router only proxies "start it" / "how's
it going" to that workflow via the GitHub REST API — nothing runs locally.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse

from ..config import settings
from ..deps import require_admin

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/ml",
    tags=["admin-ml"],
    dependencies=[Depends(require_admin)],
)

_WORKFLOW_FILE = "ml-training.yml"
_GITHUB_API = "https://api.github.com"

# Mirrors every `workflow_dispatch.inputs` default in
# .github/workflows/ml-training.yml. GitHub Actions inputs are always
# transmitted as strings, including booleans ("true"/"false").
#
# Sent explicitly rather than omitted (letting GitHub fill in the declared
# defaults itself): omitting `inputs` entirely on a workflow with boolean
# inputs is a known trigger for the GitHub API's generic, unhelpful
# "Failed to run workflow dispatch" 500 (see GitHub community reports on
# workflow_dispatch + boolean inputs) — passing every default explicitly
# sidesteps that server-side default-filling path.
_DEFAULT_WORKFLOW_INPUTS: dict[str, str] = {
    "league": "Serie A",
    "tune": "false",
    "tune_iter": "30",
    "clusters": "6",
    "predict_next": "false",
    "evaluate_mantra": "false",
    "test_seasons": "1",
    "min_minutes": "800",
    "seed": "42",
    "log_level": "INFO",
    "output_dir": "",
    "fantavoto_csv": "",
}


def _headers() -> dict:
    if not settings.github_token:
        raise HTTPException(
            status_code=503,
            detail="API_GITHUB_TOKEN not configured on the server — cannot trigger ML training.",
        )
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _dispatch_workflow() -> None:
    url = f"{_GITHUB_API}/repos/{settings.github_repo}/actions/workflows/{_WORKFLOW_FILE}/dispatches"
    resp = requests.post(
        url,
        headers=_headers(),
        json={"ref": settings.github_default_branch, "inputs": _DEFAULT_WORKFLOW_INPUTS},
        timeout=15,
    )
    resp.raise_for_status()


def _latest_run() -> Optional[dict]:
    url = f"{_GITHUB_API}/repos/{settings.github_repo}/actions/workflows/{_WORKFLOW_FILE}/runs"
    # Only completed runs represent a finished training; a queued/in_progress
    # run at the top of the list must not masquerade as the latest done one.
    resp = requests.get(
        url,
        headers=_headers(),
        params={"per_page": 1, "status": "completed"},
        timeout=15,
    )
    resp.raise_for_status()
    runs = resp.json().get("workflow_runs", [])
    return runs[0] if runs else None


async def _latest_persisted_run() -> Optional[dict]:
    """Return the most recent run that the ML pipeline actually wrote to the
    DB — the run proven to be functional (has metrics), not merely dispatched."""
    import sqlalchemy as sa

    from ..database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        row = (await db.execute(sa.text("""
            SELECT r.run_id, r.model_name, r.trained_at, r.season_start,
                   r.status, r.git_commit
            FROM model_runs r
            JOIN model_metrics m ON m.run_id = r.run_id
            GROUP BY r.id
            ORDER BY r.trained_at DESC
            LIMIT 1
        """))).fetchone()
        if row is None:
            return None
        return dict(row._mapping)


def _map_status(run: Optional[dict]) -> dict:
    if run is None:
        return {"status": "idle"}
    gh_status = run.get("status")  # queued | in_progress | completed
    conclusion = run.get("conclusion")  # success | failure | cancelled | ... | None
    if gh_status in ("queued", "in_progress"):
        status = "running"
    elif gh_status == "completed":
        status = "completed" if conclusion == "success" else "failed"
    else:
        status = "idle"
    return {
        "status": status,
        "run_number": run.get("run_number"),
        "started_at": run.get("run_started_at"),
        "updated_at": run.get("updated_at"),
        "conclusion": conclusion,
        "html_url": run.get("html_url"),
    }


@router.post("/train", summary="Trigger the ML Training GitHub Actions workflow (admin required)")
async def trigger_training() -> ORJSONResponse:
    """Dispatch .github/workflows/ml-training.yml. Returns immediately;
    poll GET /admin/ml/train/status for progress (the workflow itself
    typically takes several minutes)."""
    try:
        await asyncio.to_thread(_dispatch_workflow)
    except HTTPException:
        raise
    except requests.HTTPError as e:
        log.exception("Failed to dispatch ML Training workflow")
        detail = (
            f"GitHub API error {e.response.status_code}: {e.response.text[:300]}"
            if e.response is not None else str(e)
        )
        raise HTTPException(status_code=502, detail=detail)
    except Exception:
        log.exception("Failed to dispatch ML Training workflow")
        raise HTTPException(status_code=500, detail="Failed to trigger training workflow. Check server logs.")

    return ORJSONResponse({"status": "triggered"})


@router.get("/train/status", summary="Status of the most recent ML training run")
async def get_training_status() -> ORJSONResponse:
    try:
        # The DB is authoritative for "which run actually counts": a run only
        # shows as the functional latest one once the ML pipeline has written
        # its metrics into model_runs. GitHub's run list can lead (queued,
        # in_progress, or even failed dispatch), so we only consult it for the
        # live "running" state and the log link.
        persisted = await _latest_persisted_run()
        gh_run = await asyncio.to_thread(_latest_run)

        status = _map_status(gh_run)
        if persisted is not None:
            status["run_id"] = persisted.get("run_id")
            status["model_name"] = persisted.get("model_name")
            status["updated_at"] = persisted.get("trained_at") or status.get("updated_at")
            status["season_start"] = persisted.get("season_start")
            status["git_commit"] = persisted.get("git_commit")
            # A persisted run is, by construction, the completed one.
            if status["status"] in ("idle", "running"):
                status["status"] = "completed"
    except HTTPException:
        raise
    except requests.HTTPError as e:
        log.exception("Failed to fetch ML Training workflow status")
        detail = (
            f"GitHub API error {e.response.status_code}: {e.response.text[:300]}"
            if e.response is not None else str(e)
        )
        raise HTTPException(status_code=502, detail=detail)
    except Exception:
        log.exception("Failed to fetch ML Training workflow status")
        raise HTTPException(status_code=502, detail="Failed to fetch training status from GitHub.")

    return ORJSONResponse(status)
