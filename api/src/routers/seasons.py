from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from ..deps import get_db
from ..models import League, Season
from ..schemas import SeasonSchema

router = APIRouter(prefix="/seasons", tags=["seasons"])


@router.get("/", response_model=list[SeasonSchema], summary="List available seasons")
async def list_seasons(
    league: Optional[str] = Query(None, description="Filter by league name (partial match)"),
    db: AsyncSession = Depends(get_db),
) -> list[SeasonSchema]:
    q = select(Season).options(joinedload(Season.league))
    if league:
        q = q.join(Season.league).where(League.name.ilike(f"%{league}%"))
    q = q.order_by(Season.season_start.desc())
    result = await db.execute(q)
    return [SeasonSchema.model_validate(s) for s in result.scalars().unique().all()]


@router.get("/{season_id}", response_model=SeasonSchema, summary="Get a season by ID")
async def get_season(season_id: int, db: AsyncSession = Depends(get_db)) -> SeasonSchema:
    result = await db.execute(
        select(Season)
        .options(joinedload(Season.league))
        .where(Season.id == season_id)
    )
    season = result.scalar_one_or_none()
    if season is None:
        raise HTTPException(status_code=404, detail="Season not found")
    return SeasonSchema.model_validate(season)
