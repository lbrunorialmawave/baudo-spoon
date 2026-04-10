from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..models import League
from ..schemas import LeagueSchema

router = APIRouter(prefix="/leagues", tags=["leagues"])


@router.get("/", response_model=list[LeagueSchema], summary="List all available leagues")
async def list_leagues(db: AsyncSession = Depends(get_db)) -> list[LeagueSchema]:
    result = await db.execute(select(League).order_by(League.name))
    return [LeagueSchema.model_validate(r) for r in result.scalars().all()]


@router.get("/{league_id}", response_model=LeagueSchema, summary="Get a league by ID")
async def get_league(league_id: int, db: AsyncSession = Depends(get_db)) -> LeagueSchema:
    result = await db.execute(select(League).where(League.id == league_id))
    league = result.scalar_one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail="League not found")
    return LeagueSchema.model_validate(league)
