"""Matchday status router — live player availability information.

Endpoints
---------
GET /matchday/status                  — All players with matchday status
GET /matchday/status/{fantacalcio_id} — Single player matchday status
GET /matchday/injured                 — Only injured players
GET /matchday/suspended               — Only suspended players
GET /matchday/consigliati             — Recommended starters (probability >= 70%)
POST /matchday/scrape                 — Trigger scraper (API key)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import sqlalchemy as sa

from ml.storage.artifact_store import ArtifactStore

from ..deps import get_db, require_admin, require_role
from .intelligence import get_artifact_store

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/matchday",
    tags=["matchday"],
)


def _matchday_placeholder_sql() -> str:
    """Return SQL selecting the most recent matchday OF THE MOST RECENT SEASON.

    ``player_matchday_status`` holds one row per (season_start, matchday); a
    plain ``MAX(matchday)`` (the old behaviour) ignores the season and can
    return a matchday from an older season once multiple seasons coexist
    (e.g. 2025 md38 and 2026 md1 → wrong 38). Anchoring to the newest season
    keeps every endpoint pointing at the live line-ups.
    """
    return (
        "SELECT matchday FROM player_matchday_status "
        "WHERE season_start = (SELECT MAX(season_start) FROM player_matchday_status) "
        "ORDER BY matchday DESC LIMIT 1"
    )


@router.get("/status", summary="All players with matchday status")
async def list_matchday_status(
    matchday: Optional[int] = Query(None, description="Filter by matchday"),
    status_filter: Optional[str] = Query(None, description="Filter by status (starter, bench, injured, ...)"),
    team: Optional[str] = Query(None, description="Filter by team"),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    query = sa.text("""
        SELECT pms.*, pq.player_name, pmr.ruolo_primario, pmr.ruoli_mantra
        FROM player_matchday_status pms
        LEFT JOIN player_quotations pq
            ON pq.fantacalcio_id = pms.fantacalcio_id
            AND pq.season_start = pms.season_start
        LEFT JOIN player_mantra_roles pmr
            ON pmr.fantacalcio_id = pms.fantacalcio_id
            AND pmr.season_start = pms.season_start
        WHERE (pms.matchday = :matchday OR :matchday IS NULL)
          AND (pms.status = :status_filter OR :status_filter IS NULL)
          AND (pms.team = :team OR :team IS NULL)
        ORDER BY pms.probability DESC
    """)
    # Get latest matchday if not specified
    if matchday is None:
        latest = await db.scalar(sa.text(_matchday_placeholder_sql()))
        matchday = latest or 1

    result = await db.execute(query, {
        "matchday": matchday,
        "status_filter": status_filter,
        "team": team,
    })
    rows = [dict(r._mapping) for r in result.all()]
    return ORJSONResponse({"matchday": matchday, "count": len(rows), "items": rows})


@router.get("/status/{fantacalcio_id}", summary="Single player matchday status")
async def get_player_status(
    fantacalcio_id: int,
    matchday: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    if matchday is None:
        latest = await db.scalar(sa.text(_matchday_placeholder_sql()))
        matchday = latest or 1

    result = await db.execute(
        sa.text("""
            SELECT pms.*, pq.player_name
            FROM player_matchday_status pms
            LEFT JOIN player_quotations pq
                ON pq.fantacalcio_id = pms.fantacalcio_id
                AND pq.season_start = pms.season_start
            WHERE pms.fantacalcio_id = :fid AND pms.matchday = :md
        """),
        {"fid": fantacalcio_id, "md": matchday},
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Player {fantacalcio_id} not found for matchday {matchday}")
    return ORJSONResponse(dict(row._mapping))


@router.get("/injured", summary="Only injured players")
async def list_injured(
    matchday: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    if matchday is None:
        latest = await db.scalar(sa.text(_matchday_placeholder_sql()))
        matchday = latest or 1

    result = await db.execute(
        sa.text("""
            SELECT pms.*, pq.player_name
            FROM player_matchday_status pms
            LEFT JOIN player_quotations pq
                ON pq.fantacalcio_id = pms.fantacalcio_id
                AND pq.season_start = pms.season_start
            WHERE pms.status = 'injured' AND pms.matchday = :md
            ORDER BY pms.team, pq.player_name
        """),
        {"md": matchday},
    )
    rows = [dict(r._mapping) for r in result.all()]
    return ORJSONResponse({"matchday": matchday, "count": len(rows), "items": rows})


@router.get("/suspended", summary="Only suspended players")
async def list_suspended(
    matchday: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    if matchday is None:
        latest = await db.scalar(sa.text(_matchday_placeholder_sql()))
        matchday = latest or 1

    result = await db.execute(
        sa.text("""
            SELECT pms.*, pq.player_name
            FROM player_matchday_status pms
            LEFT JOIN player_quotations pq
                ON pq.fantacalcio_id = pms.fantacalcio_id
                AND pq.season_start = pms.season_start
            WHERE pms.status = 'suspended' AND pms.matchday = :md
            ORDER BY pms.team, pq.player_name
        """),
        {"md": matchday},
    )
    rows = [dict(r._mapping) for r in result.all()]
    return ORJSONResponse({"matchday": matchday, "count": len(rows), "items": rows})


@router.get("/consigliati", summary="Recommended starters")
async def list_consigliati(
    matchday: Optional[int] = Query(None),
    min_probability: int = Query(70, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> ORJSONResponse:
    """Return recommended players: probability >= min_probability, ordered by FP_Mantra.

    Requires MANTRA results to be available for FP_Mantra sorting.
    Falls back to probability-only sorting if MANTRA data is absent.
    """
    if matchday is None:
        latest = await db.scalar(sa.text(_matchday_placeholder_sql()))
        matchday = latest or 1

    result = await db.execute(
        sa.text("""
            SELECT pms.*, pq.player_name, pmr.ruolo_primario
            FROM player_matchday_status pms
            LEFT JOIN player_quotations pq
                ON pq.fantacalcio_id = pms.fantacalcio_id
                AND pq.season_start = pms.season_start
            LEFT JOIN player_mantra_roles pmr
                ON pmr.fantacalcio_id = pms.fantacalcio_id
                AND pmr.season_start = pms.season_start
            WHERE pms.matchday = :md AND pms.probability >= :min_prob
            ORDER BY pms.probability DESC
        """),
        {"md": matchday, "min_prob": min_probability},
    )
    rows = [dict(r._mapping) for r in result.all()]

    # Try to enrich with MANTRA scores — season resolved from the latest
    # imported quotations, read via ArtifactStore (local disk → R2), same
    # source-of-truth path used by every other MANTRA reader in this repo.
    season_start = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
    mantra_map: dict = {}
    if season_start is not None:
        mantra_data = artifact_store.load_json(f"mantra_results_{season_start}.json")
        if mantra_data is not None:
            for p in mantra_data.get("players", []):
                mantra_map[p["fantacalcio_id"]] = p

    for row in rows:
        if row["fantacalcio_id"] in mantra_map:
            mp = mantra_map[row["fantacalcio_id"]]
            row["fp_mantra"] = mp.get("FP_Mantra")
            row["vr"] = mp.get("VR")
            row["fase7"] = mp.get("Fase7")

    # Sort by FP_Mantra descending (if available), else by probability
    rows.sort(key=lambda r: r.get("fp_mantra", 0) or r.get("probability", 0), reverse=True)

    return ORJSONResponse({"matchday": matchday, "min_probability": min_probability, "count": len(rows), "items": rows})


@router.post("/scrape", summary="Trigger scraper (admin required)", dependencies=[Depends(require_admin)])
async def trigger_matchday_scrape(
    matchday: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Scrape probabili formazioni and persist to DB."""
    try:
        sync_url = str(db.bind.url).replace("+asyncpg://", "+psycopg2://").replace("?ssl=", "?sslmode=").replace("&ssl=", "&sslmode=")
        from scraper.probabili_formazioni import scrape, persist

        records = scrape(matchday=matchday)
        n = persist(records, sync_url)
        return ORJSONResponse({"status": "ok", "records": n})
    except Exception as e:
        log.exception("Matchday scrape failed")
        raise HTTPException(status_code=500, detail=str(e))
