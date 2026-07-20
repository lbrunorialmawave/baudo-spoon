"""Model metrics history router.

Endpoints
---------
GET /model-metrics/runs            — List pipeline runs with metrics (paginated)
GET /model-metrics/history         — Time-series for a specific metric
GET /model-metrics/compare         — Side-by-side comparison of two runs
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, verify_api_key

router = APIRouter(
    prefix="/model-metrics",
    tags=["model-metrics"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("/runs", response_class=ORJSONResponse, summary="List pipeline runs")
async def list_runs(
    model_name: str | None = Query(None, description="Filter by model name"),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    rows = (await db.execute(text("""
        SELECT
            r.run_id, r.model_name, r.trained_at, r.season_start,
            r.git_commit, r.status,
            COALESCE(
                json_agg(
                    json_build_object(
                        'metric', m.metric_name,
                        'value',  m.metric_value,
                        'split',  m.split
                    ) ORDER BY m.metric_name, m.split
                ) FILTER (WHERE m.id IS NOT NULL),
                '[]'::json
            ) AS metrics
        FROM model_runs r
        LEFT JOIN model_metrics m ON m.run_id = r.run_id
        WHERE (:model_name IS NULL OR r.model_name = :model_name)
        GROUP BY r.id
        ORDER BY r.trained_at DESC
        LIMIT :limit OFFSET :offset
    """), {"model_name": model_name, "limit": limit, "offset": offset})).fetchall()

    return ORJSONResponse({
        "items": [dict(r._mapping) for r in rows],
        "offset": offset,
        "limit": limit,
    })


@router.get("/history", response_class=ORJSONResponse, summary="Metric time-series")
async def metrics_history(
    metric: str = Query(default="rmse", description="Metric name, e.g. rmse, mae, r2"),
    split: str = Query(default="test", description="Split, e.g. test, backtest"),
    model_name: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    rows = (await db.execute(text("""
        SELECT r.run_id, r.trained_at, r.model_name, r.status, m.metric_value
        FROM model_metrics m
        JOIN model_runs r ON r.run_id = m.run_id
        WHERE m.metric_name = :metric
          AND m.split       = :split
          AND (:model_name IS NULL OR r.model_name = :model_name)
        ORDER BY r.trained_at ASC
    """), {"metric": metric, "split": split, "model_name": model_name})).fetchall()

    return ORJSONResponse([dict(r._mapping) for r in rows])


@router.get("/compare", response_class=ORJSONResponse, summary="Compare two runs")
async def compare_runs(
    run_a: str = Query(..., description="First run_id"),
    run_b: str = Query(..., description="Second run_id"),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    rows = (await db.execute(text("""
        SELECT
            r.run_id, r.model_name, r.trained_at, r.status,
            r.hyperparams, r.dependencies,
            COALESCE(
                json_agg(
                    json_build_object(
                        'metric', m.metric_name,
                        'value',  m.metric_value,
                        'split',  m.split
                    ) ORDER BY m.metric_name, m.split
                ) FILTER (WHERE m.id IS NOT NULL),
                '[]'::json
            ) AS metrics
        FROM model_runs r
        LEFT JOIN model_metrics m ON m.run_id = r.run_id
        WHERE r.run_id = ANY(:run_ids)
        GROUP BY r.id
        ORDER BY r.trained_at ASC
    """), {"run_ids": [run_a, run_b]})).fetchall()

    by_run: dict[str, Any] = {r.run_id: dict(r._mapping) for r in rows}
    return ORJSONResponse({"run_a": by_run.get(run_a), "run_b": by_run.get(run_b)})
