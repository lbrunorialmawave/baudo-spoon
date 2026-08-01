"""Expert ratings router — third-party opinions overlay.

Endpoints
---------
GET  /experts/ratings              — List expert ratings (paginated)
GET  /experts/ratings/{player_id}  — Ratings for a specific player
POST /experts/ratings              — Add or update an expert rating (API key)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..deps import get_db, require_role

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/experts",
    tags=["experts"],
)


class ExpertRatingCreate(BaseModel):
    player_id: str
    source: str = "gruppo_esperti"
    expert_name: Optional[str] = None
    rating: Optional[int] = None  # 1-5 stars
    comment: Optional[str] = None
    matchday: Optional[int] = None
    season_start: int
    url: Optional[str] = None


@router.get("/ratings", summary="List expert ratings")
async def list_expert_ratings(
    player_id: Optional[str] = Query(None, description="Filter by player ID"),
    source: Optional[str] = Query(None, description="Filter by source"),
    season_start: Optional[int] = Query(None, description="Filter by season"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    import sqlalchemy as sa

    conditions = ["1=1"]
    params: dict = {}
    if player_id:
        conditions.append("player_id = :player_id")
        params["player_id"] = player_id
    if source:
        conditions.append("source = :source")
        params["source"] = source
    if season_start:
        conditions.append("season_start = :season_start")
        params["season_start"] = season_start

    where = " AND ".join(conditions)

    count = await db.scalar(
        sa.text(f"SELECT COUNT(*) FROM expert_ratings WHERE {where}"),
        params,
    )
    result = await db.execute(
        sa.text(f"""
            SELECT * FROM expert_ratings
            WHERE {where}
            ORDER BY season_start DESC, id DESC
            LIMIT :size OFFSET :offset
        """),
        {**params, "size": size, "offset": (page - 1) * size},
    )
    rows = [dict(r._mapping) for r in result.all()]

    return ORJSONResponse({
        "total": int(count or 0),
        "page": page,
        "size": size,
        "items": rows,
    })


@router.get("/ratings/by-fotmob/{player_fotmob_id}", summary="Ratings for a player by FotMob ID")
async def get_player_expert_ratings_by_fotmob(
    player_fotmob_id: int,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Look up ratings via player_id_map, since expert_ratings keys players
    by the scraper's own id space (``fc-{fantacalcio_id}``) while the rest
    of the frontend addresses players by FotMob id."""
    import sqlalchemy as sa

    result = await db.execute(
        sa.text("""
            SELECT er.* FROM expert_ratings er
            JOIN player_id_map pim ON er.player_id = 'fc-' || pim.fantacalcio_id::text
            WHERE pim.player_fotmob_id = :fid
            ORDER BY er.season_start DESC, er.matchday DESC
        """),
        {"fid": player_fotmob_id},
    )
    rows = [dict(r._mapping) for r in result.all()]

    ratings = [r["rating"] for r in rows if r.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else None

    return ORJSONResponse({
        "player_fotmob_id": player_fotmob_id,
        "total_ratings": len(rows),
        "average_rating": avg_rating,
        "ratings": rows,
    })


@router.get("/ratings/{player_id}", summary="Ratings for a specific player")
async def get_player_expert_ratings(
    player_id: str,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    import sqlalchemy as sa

    result = await db.execute(
        sa.text("""
            SELECT * FROM expert_ratings
            WHERE player_id = :pid
            ORDER BY season_start DESC, matchday DESC
        """),
        {"pid": player_id},
    )
    rows = [dict(r._mapping) for r in result.all()]

    # Compute average rating
    ratings = [r["rating"] for r in rows if r.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else None

    return ORJSONResponse({
        "player_id": player_id,
        "total_ratings": len(rows),
        "average_rating": avg_rating,
        "ratings": rows,
    })


@router.post("/ratings", summary="Add or update an expert rating", dependencies=[Depends(require_role("member"))])
async def create_expert_rating(
    body: ExpertRatingCreate,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    import sqlalchemy as sa

    # Validate rating range
    if body.rating is not None and (body.rating < 1 or body.rating > 5):
        raise HTTPException(status_code=422, detail="Rating must be between 1 and 5")

    await db.execute(
        sa.text("""
            INSERT INTO expert_ratings
                (player_id, source, expert_name, rating, comment, matchday, season_start, url, scraped_at)
            VALUES
                (:player_id, :source, :expert_name, :rating, :comment, :matchday, :season_start, :url, :scraped_at)
            ON CONFLICT (player_id, source, expert_name, matchday) DO UPDATE SET
                rating = EXCLUDED.rating,
                comment = EXCLUDED.comment,
                url = EXCLUDED.url,
                scraped_at = EXCLUDED.scraped_at
        """),
        {
            "player_id": body.player_id,
            "source": body.source,
            "expert_name": body.expert_name,
            "rating": body.rating,
            "comment": body.comment,
            "matchday": body.matchday,
            "season_start": body.season_start,
            "url": body.url,
            "scraped_at": datetime.utcnow(),
        },
    )
    await db.commit()

    return ORJSONResponse({"status": "ok", "detail": "Rating saved"})
