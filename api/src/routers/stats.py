from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from ..deps import get_db
from ..models import League, PlayerSeasonStat, Season, TeamSeasonStat
from ..schemas import (
    PaginatedResponse,
    PlayerSeasonStatSchema,
    TeamSeasonStatSchema,
)

router = APIRouter(prefix="/stats", tags=["stats"])

_StatType = Literal["players", "teams"]


# ── Player stats ───────────────────────────────────────────────────────────────


@router.get(
    "/players/categories",
    response_model=list[str],
    summary="List all distinct player stat categories",
    description="Returns sorted category slugs, optionally filtered by league or season.",
    responses={200: {"description": "Sorted list of category slugs"}},
)
async def list_player_categories(
    league: Optional[str] = Query(None, description="Filter by league name (partial)"),
    season: Optional[int] = Query(None, description="Filter by season start year"),
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    q = (
        select(PlayerSeasonStat.stat_category)
        .join(PlayerSeasonStat.season)
        .join(Season.league)
        .distinct()
    )
    if league:
        q = q.where(League.name.ilike(f"%{league}%"))
    if season is not None:
        q = q.where(Season.season_start == season)
    q = q.order_by(PlayerSeasonStat.stat_category)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.get(
    "/players",
    response_model=PaginatedResponse[PlayerSeasonStatSchema],
    summary="Query player season ranking stats",
    description="Paginated player stats with filters on league, season, category, player name, and team.",
    responses={200: {"description": "Paginated player stat records"}},
)
async def list_player_stats(
    league: Optional[str] = Query(None, description="Filter by league name (partial)"),
    season: Optional[int] = Query(None, description="Filter by season start year (e.g. 2024)"),
    stat_category: Optional[str] = Query(None, description="Exact stat category slug (e.g. 'goals')"),
    player: Optional[str] = Query(None, description="Filter by player name (partial)"),
    team: Optional[str] = Query(None, description="Filter by team name (partial)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    size: int = Query(50, ge=1, le=200, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[PlayerSeasonStatSchema]:
    base_joins = (
        select(PlayerSeasonStat)
        .join(PlayerSeasonStat.season)
        .join(Season.league)
        .options(contains_eager(PlayerSeasonStat.season).contains_eager(Season.league))
    )
    count_q = (
        select(func.count(PlayerSeasonStat.id))
        .join(PlayerSeasonStat.season)
        .join(Season.league)
    )

    filters = []
    if league:
        filters.append(League.name.ilike(f"%{league}%"))
    if season is not None:
        filters.append(Season.season_start == season)
    if stat_category:
        filters.append(PlayerSeasonStat.stat_category == stat_category)
    if player:
        filters.append(PlayerSeasonStat.player_name.ilike(f"%{player}%"))
    if team:
        filters.append(PlayerSeasonStat.team_name.ilike(f"%{team}%"))

    total = await db.scalar(count_q.where(*filters)) or 0
    result = await db.execute(
        base_joins.where(*filters)
        .order_by(PlayerSeasonStat.stat_category, PlayerSeasonStat.rank)
        .offset((page - 1) * size)
        .limit(size)
    )
    items = result.scalars().all()

    return PaginatedResponse[PlayerSeasonStatSchema](
        total=total,
        page=page,
        size=size,
        items=[PlayerSeasonStatSchema.model_validate(i) for i in items],
    )


@router.get(
    "/players/{player_id}",
    response_model=list[PlayerSeasonStatSchema],
    summary="Get all stats for a specific player across seasons",
    description="Returns all stat rows for a player identified by their FotMob player ID.",
    responses={200: {"description": "All player stats across seasons"}},
)
async def get_player_stats(
    player_id: int,
    league: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> list[PlayerSeasonStatSchema]:
    q = (
        select(PlayerSeasonStat)
        .join(PlayerSeasonStat.season)
        .join(Season.league)
        .options(contains_eager(PlayerSeasonStat.season).contains_eager(Season.league))
        .where(PlayerSeasonStat.player_fotmob_id == player_id)
    )
    if league:
        q = q.where(League.name.ilike(f"%{league}%"))
    q = q.order_by(Season.season_start.desc(), PlayerSeasonStat.stat_category)
    result = await db.execute(q)
    return [PlayerSeasonStatSchema.model_validate(r) for r in result.scalars().all()]


# ── Team stats ─────────────────────────────────────────────────────────────────


@router.get(
    "/teams/categories",
    response_model=list[str],
    summary="List all distinct team stat categories",
    responses={200: {"description": "Sorted list of category slugs"}},
)
async def list_team_categories(
    league: Optional[str] = Query(None),
    season: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    q = (
        select(TeamSeasonStat.stat_category)
        .join(TeamSeasonStat.season)
        .join(Season.league)
        .distinct()
    )
    if league:
        q = q.where(League.name.ilike(f"%{league}%"))
    if season is not None:
        q = q.where(Season.season_start == season)
    q = q.order_by(TeamSeasonStat.stat_category)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.get(
    "/teams",
    response_model=PaginatedResponse[TeamSeasonStatSchema],
    summary="Query team season ranking stats",
    description="Paginated team stats with optional filters on league, season, category, and team name.",
    responses={200: {"description": "Paginated team stat records"}},
)
async def list_team_stats(
    league: Optional[str] = Query(None),
    season: Optional[int] = Query(None, description="Filter by season start year (e.g. 2024)"),
    stat_category: Optional[str] = Query(None),
    team: Optional[str] = Query(None, description="Filter by team name (partial)"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[TeamSeasonStatSchema]:
    base_joins = (
        select(TeamSeasonStat)
        .join(TeamSeasonStat.season)
        .join(Season.league)
        .options(contains_eager(TeamSeasonStat.season).contains_eager(Season.league))
    )
    count_q = (
        select(func.count(TeamSeasonStat.id))
        .join(TeamSeasonStat.season)
        .join(Season.league)
    )

    filters = []
    if league:
        filters.append(League.name.ilike(f"%{league}%"))
    if season is not None:
        filters.append(Season.season_start == season)
    if stat_category:
        filters.append(TeamSeasonStat.stat_category == stat_category)
    if team:
        filters.append(TeamSeasonStat.team_name.ilike(f"%{team}%"))

    total = await db.scalar(count_q.where(*filters)) or 0
    result = await db.execute(
        base_joins.where(*filters)
        .order_by(TeamSeasonStat.stat_category, TeamSeasonStat.rank)
        .offset((page - 1) * size)
        .limit(size)
    )
    items = result.scalars().all()

    return PaginatedResponse[TeamSeasonStatSchema](
        total=total,
        page=page,
        size=size,
        items=[TeamSeasonStatSchema.model_validate(i) for i in items],
    )


@router.get(
    "/teams/{team_id}",
    response_model=list[TeamSeasonStatSchema],
    summary="Get all stats for a specific team across seasons",
    description="Returns all stat rows for a team identified by their FotMob team ID.",
    responses={200: {"description": "All team stats across seasons"}},
)
async def get_team_stats(
    team_id: int,
    league: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> list[TeamSeasonStatSchema]:
    q = (
        select(TeamSeasonStat)
        .join(TeamSeasonStat.season)
        .join(Season.league)
        .options(contains_eager(TeamSeasonStat.season).contains_eager(Season.league))
        .where(TeamSeasonStat.team_fotmob_id == team_id)
    )
    if league:
        q = q.where(League.name.ilike(f"%{league}%"))
    q = q.order_by(Season.season_start.desc(), TeamSeasonStat.stat_category)
    result = await db.execute(q)
    return [TeamSeasonStatSchema.model_validate(r) for r in result.scalars().all()]
