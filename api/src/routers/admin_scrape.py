"""Admin router: scraper management and data health checks.

Endpoints
---------
POST /admin/scrape/snai           — Trigger Snai odds scraper (blocked from datacenter IPs — see odds-api)
POST /admin/scrape/odds-api       — Trigger The Odds API winner-odds scraper (snai replacement)
POST /admin/scrape/probabili      — Trigger probabili formazioni scraper
POST /admin/scrape/esperti        — Trigger Gruppo Esperti ratings scraper
POST /admin/scrape/quotazioni     — Re-import listoni XLSX
POST /admin/scrape/foreign-stats  — Fetch career stats for players missing Serie A history
GET  /admin/scrape/status         — Status of all scrapers
GET  /admin/scrape/logs/{name}    — Last execution log
GET  /admin/data-health           — Data coverage overview for all sources
GET  /admin/data-health/{source}  — Detailed coverage for a specific source
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..deps import get_db, require_admin

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


def _to_sync_url(url: str) -> str:
    """Convert an async SQLAlchemy DSN to a sync psycopg2 one for scraper compatibility."""
    return (
        url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgres+asyncpg://", "postgres+psycopg2://")
        .replace("?ssl=", "?sslmode=")
        .replace("&ssl=", "&sslmode=")
    )


# ── Scraper triggers ─────────────────────────────────────────────────────────


@router.post("/scrape/snai", summary="Trigger Snai odds scraper")
async def trigger_snai(
    season_start: Optional[int] = Query(None, description="Season start year"),
) -> ORJSONResponse:
    try:
        # Don't use get_db dependency — create sync connection directly for scraper compatibility
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_snai] Starting scrape")
        from scraper.snai_odds import scrape, persist
        records = scrape(season_start=season_start)
        n = persist(records, sync_url)
        return ORJSONResponse({"scraper": "snai", "records": n, "status": "ok"})
    except Exception:
        log.exception("Snai scraper failed")
        raise HTTPException(status_code=500, detail="Snai scraper failed. Check server logs.")


@router.post("/scrape/odds-api", summary="Trigger The Odds API winner-odds scraper")
async def trigger_odds_api(
    season_start: Optional[int] = Query(None, description="Season start year"),
) -> ORJSONResponse:
    if not settings.odds_api_key:
        raise HTTPException(
            status_code=400,
            detail="ODDS_API_KEY not configured. Set API_ODDS_API_KEY on the server.",
        )
    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_odds_api] Starting scrape")
        from scraper.odds_api import scrape, persist
        records = scrape(api_key=settings.odds_api_key, season_start=season_start)
        n = persist(records, sync_url)
        return ORJSONResponse({"scraper": "odds-api", "records": n, "status": "ok"})
    except Exception:
        log.exception("The Odds API scraper failed")
        raise HTTPException(status_code=500, detail="The Odds API scraper failed. Check server logs.")


@router.post("/scrape/probabili", summary="Trigger probabili formazioni scraper")
async def trigger_probabili(
    matchday: Optional[int] = Query(None, description="Current matchday"),
) -> ORJSONResponse:
    try:
        # Don't use get_db dependency — create sync connection directly for scraper compatibility
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_probabili] Starting scrape")
        from scraper.probabili_formazioni import scrape, persist
        records = scrape(matchday=matchday)
        n = persist(records, sync_url)
        return ORJSONResponse({"scraper": "probabili", "records": n, "status": "ok"})
    except Exception:
        log.exception("Probabili formazioni scraper failed")
        raise HTTPException(status_code=500, detail="Probabili formazioni scraper failed. Check server logs.")


@router.post("/scrape/esperti", summary="Trigger Gruppo Esperti ratings scraper")
async def trigger_esperti(
    season_start: Optional[int] = Query(None, description="Season start year"),
    team: Optional[str] = Query(None, description="Only scrape one team (e.g. 'Inter')"),
) -> ORJSONResponse:
    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_esperti] Starting scrape")
        from scraper.gruppo_esperti import scrape, persist
        players = scrape(team_filter=team)
        # season_start=None resolves internally to the latest season present
        # in player_quotations, rather than guessing from the calendar date.
        n, resolved_season = persist(players, sync_url, season_start)
        return ORJSONResponse({
            "scraper": "esperti",
            "season_start": resolved_season,
            "scraped": len(players),
            "records": n,
            "unmatched": len(players) - n,
            "status": "ok",
        })
    except Exception:
        log.exception("Gruppo Esperti scraper failed")
        raise HTTPException(status_code=500, detail="Gruppo Esperti scraper failed. Check server logs.")


@router.post("/scrape/quotazioni", summary="Re-import listoni XLSX")
async def trigger_quotazioni(
    quotazioni_dir: str = Query("./quotazioni", description="Directory with XLSX files"),
) -> ORJSONResponse:
    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info(f"[trigger_quotazioni] Using sync_url: {sync_url}")
        import subprocess
        result = subprocess.run(
            ["python", "-m", "ml.data.import_quotations",
             "--quotazioni-dir", quotazioni_dir,
             "--db-url", sync_url],
            capture_output=True, text=True, timeout=300,
        )
        return ORJSONResponse({
            "scraper": "quotazioni",
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
        })
    except Exception:
        log.exception("Quotazioni import failed")
        raise HTTPException(status_code=500, detail="Quotazioni import failed. Check server logs.")


_FOREIGN_STATS_CANDIDATES_SQL = sa.text("""
    SELECT DISTINCT pim.player_fotmob_id, pq.player_name
    FROM player_quotations pq
    JOIN player_id_map pim
        ON pim.fantacalcio_id = pq.fantacalcio_id
        AND pim.season_start = pq.season_start
    LEFT JOIN player_season_aggregates pss_cur
        ON pss_cur.fantacalcio_id = pim.player_fotmob_id::bigint
        AND pss_cur.season_start = pq.season_start
    LEFT JOIN player_season_aggregates pss_prev
        ON pss_prev.fantacalcio_id = pim.player_fotmob_id::bigint
        AND pss_prev.season_start = pq.season_start - 1
    LEFT JOIN player_latest_stats_any_league pss_any
        ON pss_any.fantacalcio_id = pim.player_fotmob_id::bigint
    WHERE pq.season_start = :season_start
      AND pim.player_fotmob_id IS NOT NULL
      AND pss_cur.fantacalcio_id IS NULL
      AND pss_prev.fantacalcio_id IS NULL
      AND (:force OR pss_any.fantacalcio_id IS NULL)
""")


@router.post(
    "/scrape/foreign-stats",
    summary="Fetch career stats for players missing Serie A history",
)
async def trigger_foreign_stats(
    force: bool = Query(
        False,
        description="Re-fetch even for players who already have some foreign-league data",
    ),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Targeted per-player fallback for MANTRA's neo-arrivo handling.

    Finds players in the current Serie A listino who have no Serie A
    performance history (current or prior season) and, unless already
    covered, fetches their most recent season anywhere via a single
    lightweight HTTP request per player (see
    scraper/src/player_career_scraper.py) — no bulk league scraping.
    """
    latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
    if latest is None:
        raise HTTPException(status_code=400, detail="No quotations imported yet.")

    result = await db.execute(_FOREIGN_STATS_CANDIDATES_SQL, {"season_start": latest, "force": force})
    candidates = {row.player_fotmob_id: row.player_name for row in result.all()}

    if not candidates:
        return ORJSONResponse({
            "scraper": "foreign-stats", "status": "ok",
            "candidates": 0, "fetched": 0, "persisted": 0,
        })

    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_foreign_stats] %d candidate(s), force=%s", len(candidates), force)
        from scraper.src.player_career_scraper import fetch_and_persist_players

        fetched, persisted = fetch_and_persist_players(candidates, sync_url)
        return ORJSONResponse({
            "scraper": "foreign-stats",
            "status": "ok",
            "candidates": len(candidates),
            "fetched": fetched,
            "persisted": persisted,
            "unresolved": len(candidates) - fetched,
        })
    except Exception:
        log.exception("Foreign career-stats fetch failed")
        raise HTTPException(status_code=500, detail="Foreign career-stats fetch failed. Check server logs.")


# ── Data Health ──────────────────────────────────────────────────────────────


@router.get("/data-health", summary="Data coverage overview")
async def get_data_health(
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Return coverage status for all data sources."""
    import sqlalchemy as sa

    sources: list[dict] = []

    # 1. ID Mapping health
    id_total = await db.scalar(sa.text("SELECT COUNT(*) FROM player_id_map"))
    id_matched = await db.scalar(
        sa.text("SELECT COUNT(*) FROM player_id_map WHERE player_fotmob_id IS NOT NULL")
    )
    id_unmatched = int(id_total) - int(id_matched) if id_total else 0
    match_rate = (int(id_matched) / int(id_total) * 100) if int(id_total) > 0 else 0
    sources.append({
        "name": "id_mapping",
        "total_rows": int(id_total or 0),
        "matched": int(id_matched or 0),
        "unmatched": id_unmatched,
        "match_rate_pct": round(match_rate, 1),
        "status": "ok" if match_rate >= 95 else "warning",
    })

    # 2. MANTRA roles
    mantra_count = await db.scalar(sa.text("SELECT COUNT(*) FROM player_mantra_roles"))
    sources.append({
        "name": "mantra_roles",
        "total_rows": int(mantra_count or 0),
        "status": "ok" if int(mantra_count or 0) > 0 else "missing",
    })

    # 3. Matchday status
    md_count = await db.scalar(
        sa.text("SELECT COUNT(*) FROM player_matchday_status")
    )
    md_latest = await db.scalar(
        sa.text("SELECT MAX(matchday) FROM player_matchday_status")
    )
    sources.append({
        "name": "matchday_status",
        "total_rows": int(md_count or 0),
        "latest_matchday": md_latest,
        "status": "ok" if int(md_count or 0) > 0 else "missing",
    })

    # 4. Team winner odds (either source — snai is blocked from datacenter IPs,
    # odds_api is the intended replacement, but both write the same table/shape).
    snai_count = await db.scalar(
        sa.text("SELECT COUNT(*) FROM team_season_odds WHERE source IN ('snai', 'odds_api')")
    )
    sources.append({
        "name": "snai_odds",
        "total_rows": int(snai_count or 0),
        "status": "ok" if int(snai_count or 0) > 0 else "missing",
    })

    # 5. Expert ratings
    exp_count = await db.scalar(sa.text("SELECT COUNT(*) FROM expert_ratings"))
    sources.append({
        "name": "expert_ratings",
        "total_rows": int(exp_count or 0),
        "status": "ok" if int(exp_count or 0) > 0 else "missing",
    })

    # 6. Quotations
    q_count = await db.scalar(
        sa.text("SELECT COUNT(*) FROM player_quotations")
    )
    q_seasons = await db.scalar(
        sa.text("SELECT array_agg(DISTINCT season_start ORDER BY season_start DESC) FROM player_quotations")
    )
    sources.append({
        "name": "quotations",
        "total_rows": int(q_count or 0),
        "seasons": q_seasons,
        "status": "ok" if int(q_count or 0) > 450 else "warning",
    })

    return ORJSONResponse({"sources": sources})


@router.get("/data-health/{source}", summary="Detailed coverage for a source")
async def get_source_health(
    source: str,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Return detailed coverage info for a specific data source."""
    import sqlalchemy as sa

    if source == "id_mapping":
        rows = (await db.execute(
            sa.text("""
                SELECT match_method, COUNT(*) as cnt
                FROM player_id_map
                GROUP BY match_method
                ORDER BY cnt DESC
            """)
        )).all()
        return ORJSONResponse({
            "source": source,
            "by_method": {r.match_method: int(r.cnt) for r in rows},
        })

    if source == "matchday_status":
        rows = (await db.execute(
            sa.text("""
                SELECT status, COUNT(*) as cnt
                FROM player_matchday_status
                GROUP BY status
                ORDER BY cnt DESC
            """)
        )).all()
        return ORJSONResponse({
            "source": source,
            "by_status": {r.status: int(r.cnt) for r in rows},
        })

    if source == "mantra_roles":
        rows = (await db.execute(
            sa.text("""
                SELECT ruolo_primario, COUNT(*) as cnt
                FROM player_mantra_roles
                GROUP BY ruolo_primario
                ORDER BY cnt DESC
            """)
        )).all()
        return ORJSONResponse({
            "source": source,
            "by_role": {r.ruolo_primario: int(r.cnt) for r in rows},
        })

    return ORJSONResponse({"source": source, "detail": "No detailed breakdown available"})


@router.get("/scrape/status", summary="Status of all scrapers")
async def get_scraper_status() -> ORJSONResponse:
    """Return static status information about available scrapers."""
    scrapers = [
        {
            "name": "snai",
            "description": "Snai Serie A winner odds (blocked from datacenter IPs — see odds-api)",
            "frequency": "Pre-season + January",
            "configurable_params": ["season_start"],
        },
        {
            "name": "odds-api",
            "description": "Serie A winner odds via The Odds API (the-odds-api.com)",
            "frequency": "Pre-season + January",
            "configurable_params": ["season_start"],
        },
        {
            "name": "probabili",
            "description": "Probabili formazioni Serie A",
            "frequency": "Every matchday",
            "configurable_params": ["matchday", "url"],
        },
        {
            "name": "esperti",
            "description": "Gruppo Esperti player ratings and comments",
            "frequency": "Season start + periodic re-scrape",
            "configurable_params": ["season_start", "team"],
        },
        {
            "name": "quotazioni",
            "description": "Re-import Fantacalcio listoni XLSX",
            "frequency": "Season start",
            "configurable_params": ["quotazioni_dir"],
        },
        {
            "name": "voti",
            "description": "Re-parse Fantacalcio voti JSON",
            "frequency": "Every matchday",
            "configurable_params": ["voti_dir"],
        },
    ]
    return ORJSONResponse({"scrapers": scrapers})


@router.get("/scrape/logs/{name}", summary="Last execution log")
async def get_scraper_log(name: str) -> ORJSONResponse:
    """Return last execution log for a scraper (mock — real impl would read a log file)."""
    return ORJSONResponse({
        "name": name,
        "last_run": None,
        "status": "unknown",
        "message": "Log tracking not yet implemented. Check terminal output.",
    })
