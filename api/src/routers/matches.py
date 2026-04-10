from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from ..deps import get_db
from ..models import League, MatchStat, Season
from ..schemas import MatchStatSchema, PaginatedResponse

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get(
    "/",
    response_model=PaginatedResponse[MatchStatSchema],
    summary="List match statistics",
    description="Paginated match stats with optional filters on league, season, team, opponent, or match name.",
    responses={200: {"description": "Paginated match stat records"}},
)
async def list_matches(
    league: Optional[str] = Query(None, description="Filter by league name (partial match)"),
    season: Optional[int] = Query(None, description="Filter by season start year (e.g. 2023)"),
    team: Optional[str] = Query(None, description="Filter by team name (partial match)"),
    opponent: Optional[str] = Query(None, description="Filter by opponent name (partial match)"),
    search: Optional[str] = Query(None, description="Search match name (case-insensitive)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[MatchStatSchema]:
    filters = []
    if league:
        filters.append(League.name.ilike(f"%{league}%"))
    if season is not None:
        filters.append(Season.season_start == season)
    if team:
        filters.append(MatchStat.team.ilike(f"%{team}%"))
    if opponent:
        filters.append(MatchStat.opponent.ilike(f"%{opponent}%"))
    if search:
        filters.append(MatchStat.match_name.ilike(f"%{search}%"))

    count_q = (
        select(func.count(MatchStat.id))
        .join(MatchStat.season)
        .join(Season.league)
        .where(*filters)
    )
    total = await db.scalar(count_q) or 0

    items_q = (
        select(MatchStat)
        .join(MatchStat.season)
        .join(Season.league)
        .options(contains_eager(MatchStat.season).contains_eager(Season.league))
        .where(*filters)
        .order_by(MatchStat.match_date.desc(), MatchStat.match_name)
        .offset((page - 1) * size)
        .limit(size)
    )
    result = await db.execute(items_q)
    items = result.scalars().all()

    return PaginatedResponse[MatchStatSchema](
        total=total,
        page=page,
        size=size,
        items=[MatchStatSchema.model_validate(item) for item in items],
    )


@router.get(
    "/{match_id}",
    response_model=MatchStatSchema,
    summary="Get a match stat record by ID",
    description="Returns a single match stat record, or 404 if not found.",
    responses={
        200: {"description": "Match stat record"},
        404: {"description": "Record not found"},
    },
)
async def get_match(
    match_id: int,
    db: AsyncSession = Depends(get_db),
) -> MatchStatSchema:
    result = await db.execute(
        select(MatchStat)
        .join(MatchStat.season)
        .join(Season.league)
        .options(contains_eager(MatchStat.season).contains_eager(Season.league))
        .where(MatchStat.id == match_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Match stat record not found")
    return MatchStatSchema.model_validate(record)
