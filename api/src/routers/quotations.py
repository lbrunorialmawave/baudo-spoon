"""Routers for Fantacalcio quotations and FotMob ID-mapping data.

Public routes (``/api/v1/quotations/*``) are read-only and unauthenticated.
ID-mapping routes (``/api/v1/intelligence/id-mapping/*``) require the
``X-API-Key`` header because the mapping table contains operational
metadata used to build the training dataset.

Routes
------
GET /quotations                              — Paginated list of player quotations (DB).
GET /quotations/seasons                      — Distinct seasons present in the table.
GET /quotations/seasons/{season_start}       — Quotations for one season.
GET /quotations/players/{player_fotmob_id}   — History for a single FotMob player.
GET /quotations/stats                        — Aggregate statistics across all rows.

GET /intelligence/id-mapping                 — Paginated list of Fantacalcio↔FotMob mappings (API key).
GET /intelligence/id-mapping/{id}            — Single mapping lookup (API key).
GET /intelligence/id-mapping/stats           — Match rate + per-method counts (API key).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, verify_api_key
from ..schemas import (
    IdMappingStatsResponse,
    PlayerIdMapSchema,
    PlayerQuotationWithMappingSchema,
    QuotationStatsResponse,
)

log = logging.getLogger(__name__)

# ── Valid role / method enums (mirrors DB CHECK constraints) ─────────────────

_VALID_ROLES = {"GK", "DEF", "MID", "FWD"}
_VALID_MATCH_METHODS = {
    "exact_name_team",
    "exact_name_role",
    "exact_relaxed_role",
    "fuzzy_name",
    "manual",
    "unmatched",
}


def get_repository(request: Request) -> DataRepository:
    """Retrieve the application-scoped DataRepository from app.state.

    Quotations/ID-mapping endpoints only need the DB-touching methods, but
    we still rely on the singleton so we can extend the repository (e.g. a
    cache layer) without changing the route signatures.
    """
    repo: DataRepository | None = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Data repository not initialised")
    return repo


# ── Public quotations router ──────────────────────────────────────────────────

quotations_router = APIRouter(prefix="/quotations", tags=["quotations"])


@quotations_router.get(
    "",
    response_class=ORJSONResponse,
    summary="List Fantacalcio quotations",
    description=(
        "Returns a paginated list of player auction valuations from "
        "``player_quotations``, left-joined to ``player_id_map`` so the "
        "FotMob ID and match metadata are available when present. "
        "All filters are optional and combined with AND."
    ),
    responses={
        200: {"description": "Paginated list of quotations with mapping fields"},
        400: {"description": "Invalid query parameter (e.g. unknown role)"},
    },
)
async def list_quotations(
    season_start: Optional[int] = Query(None, ge=1990, le=2100, description="Filter by season start year"),
    role: Optional[str] = Query(None, description="Filter by canonical role (GK, DEF, MID, FWD)"),
    team: Optional[str] = Query(None, description="Filter by team name (exact match)"),
    player_fotmob_id: Optional[int] = Query(None, ge=1, description="Filter by FotMob player ID"),
    min_qt_a: Optional[int] = Query(None, ge=0, description="Minimum Qt.A value (inclusive)"),
    max_qt_a: Optional[int] = Query(None, ge=0, description="Maximum Qt.A value (inclusive)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    size: int = Query(50, ge=1, le=500, description="Items per page (max 500)"),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    if role is not None and role.upper() not in _VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of {sorted(_VALID_ROLES)} (got {role!r})",
        )
    if min_qt_a is not None and max_qt_a is not None and min_qt_a > max_qt_a:
        raise HTTPException(
            status_code=400,
            detail=f"min_qt_a ({min_qt_a}) cannot exceed max_qt_a ({max_qt_a})",
        )

    rows, total = await repo.list_quotations(
        db=db,
        season_start=season_start,
        role=role.upper() if role else None,
        team=team,
        player_fotmob_id=player_fotmob_id,
        min_qt_a=min_qt_a,
        max_qt_a=max_qt_a,
        page=page,
        size=size,
    )

    items = [PlayerQuotationWithMappingSchema(**r).model_dump(by_alias=True) for r in rows]
    return ORJSONResponse(
        content={"total": total, "page": page, "size": size, "items": items}
    )


@quotations_router.get(
    "/seasons",
    response_class=ORJSONResponse,
    summary="Distinct seasons with quotation data",
    description="Returns every ``season_start`` value present in ``player_quotations``, newest first.",
)
async def list_quotation_seasons(
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    seasons = await repo.get_quotation_seasons(db=db)
    return ORJSONResponse(content=seasons)


@quotations_router.get(
    "/seasons/{season_start}",
    response_class=ORJSONResponse,
    summary="Quotations for one season",
    description="Returns all quotations for the requested season, joined to the id-map. Not paginated.",
    responses={
        200: {"description": "All quotations for the season"},
        404: {"description": "No quotations found for the requested season"},
    },
)
async def list_quotations_by_season(
    season_start: int,
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    rows, total = await repo.list_quotations(
        db=db, season_start=season_start, page=1, size=2000
    )
    if total == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No quotations found for season_start={season_start}",
        )
    items = [PlayerQuotationWithMappingSchema(**r).model_dump(by_alias=True) for r in rows]
    return ORJSONResponse(content={"seasonStart": season_start, "total": total, "items": items})


@quotations_router.get(
    "/players/{player_fotmob_id}",
    response_class=ORJSONResponse,
    summary="Historical quotations for one FotMob player",
    description=(
        "Returns every quotation row associated with the given FotMob ID, "
        "ordered by season descending. Useful for tracking a player's price "
        "trajectory across multiple Fantacalcio seasons."
    ),
    responses={
        200: {"description": "List of historical quotations for the player"},
        404: {"description": "No quotations found for this FotMob player"},
    },
)
async def get_player_fotmob_history(
    player_fotmob_id: int,
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    rows = await repo.get_player_fotmob_history(db=db, player_fotmob_id=player_fotmob_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No quotations found for player_fotmob_id={player_fotmob_id}",
        )
    items = [PlayerQuotationWithMappingSchema(**r).model_dump(by_alias=True) for r in rows]
    return ORJSONResponse(
        content={"playerFotmobId": player_fotmob_id, "total": len(items), "items": items}
    )


@quotations_router.get(
    "/stats",
    response_class=ORJSONResponse,
    summary="Quotation aggregate statistics",
    description=(
        "Returns total row count, distinct seasons, distinct team count, "
        "per-season-per-role aggregates (mean / median / min / max Qt.A and "
        "avg FVM), and the id-mapping method coverage. Designed for dashboard "
        "cards and data-quality checks."
    ),
)
async def get_quotation_stats(
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    stats = await repo.get_quotation_stats(db=db)
    payload = QuotationStatsResponse(**stats).model_dump(by_alias=True)
    return ORJSONResponse(content=payload)


# ── ID-mapping router (API-key protected + rate-limited) ─────────────────────

id_mapping_router = APIRouter(
    prefix="/intelligence/id-mapping",
    tags=["id-mapping"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)


@id_mapping_router.get(
    "",
    response_class=ORJSONResponse,
    summary="Paginated Fantacalcio↔FotMob ID mapping",
    description=(
        "Returns the contents of the ``player_id_map`` bridge table. Use "
        "``matched_only=true`` to exclude UNMATCHED rows, or filter by "
        "``match_method`` to inspect the resolution pipeline output."
    ),
    responses={
        200: {"description": "Paginated list of mapping rows"},
        400: {"description": "Invalid query parameter"},
    },
)
async def list_id_mappings(
    season_start: Optional[int] = Query(None, ge=1990, le=2100, description="Filter by season"),
    match_method: Optional[str] = Query(
        None,
        description="Match algorithm (exact_name_team, exact_name_role, exact_relaxed_role, fuzzy_name, manual, unmatched)",
    ),
    canonical_role: Optional[str] = Query(None, description="Filter by canonical role (GK, DEF, MID, FWD)"),
    matched_only: bool = Query(False, description="When true, excludes UNMATCHED rows"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    if match_method is not None and match_method not in _VALID_MATCH_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"match_method must be one of {sorted(_VALID_MATCH_METHODS)} (got {match_method!r})",
        )
    if canonical_role is not None and canonical_role.upper() not in _VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"canonical_role must be one of {sorted(_VALID_ROLES)} (got {canonical_role!r})",
        )

    rows, total = await repo.list_id_mappings(
        db=db,
        season_start=season_start,
        match_method=match_method,
        canonical_role=canonical_role.upper() if canonical_role else None,
        matched_only=matched_only,
        page=page,
        size=size,
    )
    items = [PlayerIdMapSchema(**r).model_dump(by_alias=True) for r in rows]
    return ORJSONResponse(
        content={"total": total, "page": page, "size": size, "items": items}
    )


@id_mapping_router.get(
    "/stats",
    response_class=ORJSONResponse,
    summary="ID-mapping match-rate statistics",
    description=(
        "Returns the global match rate plus a breakdown by season and by "
        "``match_method``. Useful for monitoring the id-resolution pipeline "
        "and detecting regressions in the matching heuristics."
    ),
)
async def get_id_mapping_stats(
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    stats = await repo.get_id_mapping_stats(db=db)
    payload = IdMappingStatsResponse(**stats).model_dump(by_alias=True)
    return ORJSONResponse(content=payload)


@id_mapping_router.get(
    "/{fantacalcio_id}",
    response_class=ORJSONResponse,
    summary="Single Fantacalcio→FotMob mapping lookup",
    description=(
        "Returns the mapping row for the given ``fantacalcio_id``. If "
        "``season_start`` is provided the lookup is exact; otherwise the most "
        "recent mapping for that ID is returned."
    ),
    responses={
        200: {"description": "Mapping row"},
        404: {"description": "No mapping found for the requested ID"},
    },
)
async def get_id_mapping(
    fantacalcio_id: int,
    season_start: Optional[int] = Query(
        None, ge=1990, le=2100, description="Optional season start (defaults to most recent)"
    ),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    row = await repo.get_id_mapping(db=db, fantacalcio_id=fantacalcio_id, season_start=season_start)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No mapping found for fantacalcio_id={fantacalcio_id}"
                + (f" in season_start={season_start}" if season_start else "")
            ),
        )
    return ORJSONResponse(content=PlayerIdMapSchema(**row).model_dump(by_alias=True))
