"""Routers for ML-driven insights: player predictions and clustering intelligence.

Routes
------
GET  /predictions/players                  — Paginated player predictions (ML + DB metadata).
GET  /predictions/next-season              — Next-season projected ratings.
GET  /intelligence/clustering/players      — Full cluster membership list.
GET  /intelligence/clustering/alternatives — Low-cost player clones (requires API key).
POST /intelligence/cache/invalidate        — Evict Redis cache (requires API key).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, verify_api_key
from ..models import PlayerSeasonStat, Season
from ..schemas import (
    AlternativesResponse,
    ClusteringStatsSchema,
    LowCostAlternativeSchema,
    ModelComparisonSchema,
    NextSeasonPredictionSchema,
    PlayerClusterSchema,
    PlayerPredictionSchema,
)

log = logging.getLogger(__name__)

# ── Shared dependency ─────────────────────────────────────────────────────────


def get_repository(request: Request) -> DataRepository:
    """Retrieve the application-scoped DataRepository from app.state."""
    repo: DataRepository | None = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="ML data repository not initialised")
    return repo


# ── Predictions router (public) ───────────────────────────────────────────────

predictions_router = APIRouter(prefix="/predictions", tags=["predictions"])


@predictions_router.get(
    "/players",
    response_class=ORJSONResponse,
    summary="Paginated ML player predictions",
    description=(
        "Returns player Fantacalcio rating predictions from the latest ML run, "
        "enriched with team metadata from **PlayerSeasonStat**. "
        "Supports filtering by player name, team, and canonical role."
    ),
    responses={
        200: {"description": "Paginated prediction envelope with run metadata"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_player_predictions(
    player: Optional[str] = Query(None, description="Filter by player name (partial, case-insensitive)"),
    team: Optional[str] = Query(None, description="Filter by team name (partial, case-insensitive)"),
    role: Optional[str] = Query(None, description="Canonical role: GK, DEF, MID, FWD"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    size: int = Query(50, ge=1, le=200, description="Items per page"),
    repo: DataRepository = Depends(get_repository),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        raw = await repo.get_predictions()
        meta = await repo.get_run_metadata()
        model_comparison = await repo.get_model_comparison()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Enrich: build player_name → DB metadata lookup from the most-recent season.
    db_result = await db.execute(
        select(
            PlayerSeasonStat.player_name,
            PlayerSeasonStat.player_fotmob_id,
            PlayerSeasonStat.team_name,
        )
        .join(PlayerSeasonStat.season)
        .distinct(PlayerSeasonStat.player_fotmob_id)
        .order_by(PlayerSeasonStat.player_fotmob_id, Season.season_start.desc())
    )
    db_lookup: dict[str, dict] = {
        row.player_name: {"player_fotmob_id": row.player_fotmob_id, "team_name": row.team_name}
        for row in db_result.all()
    }

    # Merge ML records with DB metadata and apply filters.
    items: list[PlayerPredictionSchema] = []
    for r in raw:
        name: str = r.get("player_name", "")
        db_meta = db_lookup.get(name, {})
        item = PlayerPredictionSchema(
            player_name=name,
            player_fotmob_id=r.get("player_fotmob_id") or db_meta.get("player_fotmob_id"),
            team_name=r.get("team_name") or db_meta.get("team_name"),
            canonical_role=r.get("canonical_role"),
            season=r.get("season"),
            fantavoto_medio=r.get("fantavoto_medio"),
            predicted=r.get("predicted", 0.0),
        )
        if player and player.lower() not in name.lower():
            continue
        if team and (not item.team_name or team.lower() not in item.team_name.lower()):
            continue
        if role and item.canonical_role != role.upper():
            continue
        items.append(item)

    total = len(items)
    page_items = items[(page - 1) * size : page * size]

    payload = {
        "runId": meta["run_id"],
        "bestModel": meta["best_model"],
        "rolePartitioned": meta["role_partitioned"],
        "modelComparison": [ModelComparisonSchema(**m).model_dump(by_alias=True) for m in model_comparison],
        "total": total,
        "page": page,
        "size": size,
        "items": [p.model_dump(by_alias=True) for p in page_items],
    }
    return ORJSONResponse(content=payload)


@predictions_router.get(
    "/next-season",
    response_class=ORJSONResponse,
    summary="Next-season projected player ratings",
    description=(
        "Forward-projected Fantacalcio ratings generated when the pipeline was "
        "run with ``--predict-next``. Returns 404 when not available."
    ),
    responses={
        200: {"description": "List of next-season predictions"},
        404: {"description": "No next-season predictions available"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_next_season_predictions(
    player: Optional[str] = Query(None, description="Filter by player name (partial)"),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        raw = await repo.get_next_season_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not raw:
        raise HTTPException(status_code=404, detail="No next-season predictions in current artifact")

    items = [NextSeasonPredictionSchema(**r) for r in raw]
    if player:
        q = player.lower()
        items = [i for i in items if q in i.player_name.lower()]

    return ORJSONResponse(content=[i.model_dump(by_alias=True) for i in items])


# ── Intelligence router (API-key protected + rate-limited) ────────────────────

intelligence_router = APIRouter(
    prefix="/intelligence",
    tags=["intelligence"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)


@intelligence_router.get(
    "/clustering/players",
    response_class=ORJSONResponse,
    summary="Player cluster assignments",
    description=(
        "PCA-reduced cluster membership for every player in the latest ML run. "
        "Useful for building player similarity maps and visualisations."
    ),
    responses={
        200: {"description": "Paginated cluster assignments with clustering stats"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_cluster_players(
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    role: Optional[str] = Query(None, description="Canonical role: GK, DEF, MID, FWD"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        raw = await repo.get_player_clusters()
        stats = await repo.get_clustering_stats()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = [PlayerClusterSchema(**r) for r in raw]
    if cluster_id is not None:
        items = [i for i in items if i.cluster_id == cluster_id]
    if role:
        items = [i for i in items if i.canonical_role == role.upper()]

    total = len(items)
    page_items = items[(page - 1) * size : page * size]

    payload = {
        "clusteringStats": ClusteringStatsSchema(**stats).model_dump(by_alias=True),
        "total": total,
        "page": page,
        "size": size,
        "items": [p.model_dump(by_alias=True) for p in page_items],
    }
    return ORJSONResponse(content=payload)


@intelligence_router.get(
    "/clustering/alternatives",
    response_class=ORJSONResponse,
    summary="Low-cost player alternatives",
    description=(
        "For each top-percentile player (above the 80th percentile of Fantacalcio "
        "rating) returns cluster-mates from less prestigious clubs — a.k.a. "
        "'budget clones'. Filter by ``top_player_id`` to focus on a single player. "
        "Ideal for budget-constrained Fantacalcio roster construction."
    ),
    responses={
        200: {"description": "Alternatives + clustering metadata"},
        404: {"description": "No recommendations found for requested player"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_low_cost_alternatives(
    top_player_id: Optional[int] = Query(
        None, description="FotMob player ID — filter recommendations for one top player"
    ),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        recs = await repo.get_low_cost_recommendations(top_player_id=top_player_id)
        stats = await repo.get_clustering_stats()
        clusters = await repo.get_player_clusters()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if top_player_id is not None and not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for player_id={top_player_id}",
        )

    response = AlternativesResponse(
        clustering_stats=ClusteringStatsSchema(**stats),
        player_clusters=[PlayerClusterSchema(**c) for c in clusters],
        low_cost_recommendations=[LowCostAlternativeSchema(**r) for r in recs],
    )
    return ORJSONResponse(content=response.model_dump(by_alias=True))


@intelligence_router.post(
    "/cache/invalidate",
    summary="Invalidate ML result cache",
    description=(
        "Evicts Redis-cached ML artifact entries. "
        "Call this after deploying a new ML pipeline run to ensure stale data is not served."
    ),
    responses={200: {"description": "Cache invalidated successfully"}},
)
async def invalidate_cache(
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    await repo.invalidate_cache()
    return ORJSONResponse(content={"detail": "Cache invalidated"})
