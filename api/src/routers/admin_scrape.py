"""Admin router: scraper management and data health checks.

Endpoints
---------
POST /admin/scrape/probabili         — Trigger probabili formazioni scraper
POST /admin/scrape/esperti           — Trigger Gruppo Esperti ratings scraper
POST /admin/scrape/quotazioni        — Re-import listoni XLSX (+ auto resolve-unmatched + foreign-stats)
POST /admin/scrape/foreign-stats     — Fetch career stats for players missing Serie A history
POST /admin/scrape/resolve-unmatched — Retry FotMob resolution for unmatched players
GET  /admin/scrape/status            — Status of all scrapers
GET  /admin/scrape/logs/{name}       — Last execution log
GET  /admin/data-health              — Data coverage overview for all sources
GET  /admin/data-health/{source}     — Detailed coverage for a specific source
"""

from __future__ import annotations

import logging
from pathlib import Path

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


@router.post("/scrape/probabili", summary="Trigger probabili formazioni scraper")
async def trigger_probabili(
    matchday: int | None = Query(None, description="Current matchday"),
) -> ORJSONResponse:
    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_probabili] Starting scrape")
        from scraper.probabili_formazioni import persist, scrape
        records = scrape(matchday=matchday)
        n = persist(records, sync_url)
        return ORJSONResponse({"scraper": "probabili", "records": n, "status": "ok"})
    except Exception:
        log.exception("Probabili formazioni scraper failed")
        raise HTTPException(status_code=500, detail="Probabili formazioni scraper failed. Check server logs.")


@router.post("/scrape/esperti", summary="Trigger Gruppo Esperti ratings scraper")
async def trigger_esperti(
    season_start: int | None = Query(None, description="Season start year"),
    team: str | None = Query(None, description="Only scrape one team (e.g. 'Inter')"),
    index_url: str | None = Query(
        None,
        description=(
            "Override the season index/listing URL (viewtopic.php or "
            "viewforum.php) without a redeploy, e.g. if the forum "
            "reorganizes again. Defaults to gruppo_esperti.INDEX_URL."
        ),
    ),
) -> ORJSONResponse:
    try:
        sync_url = _to_sync_url(settings.database_url)
        log.info("[trigger_esperti] Starting scrape")
        from scraper.gruppo_esperti import INDEX_URL, persist, scrape
        players = scrape(index_url=index_url or INDEX_URL, team_filter=team)
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


async def _run_foreign_stats(
    db: AsyncSession,
    *,
    force: bool = False,
) -> dict:
    """Shared logic for targeted career-stats fetch of neo-arrivi."""
    latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
    if latest is None:
        return {
            "status": "skipped",
            "reason": "no_quotations",
            "candidates": 0,
            "fetched": 0,
            "persisted": 0,
        }

    result = await db.execute(
        _FOREIGN_STATS_CANDIDATES_SQL,
        {"season_start": latest, "force": force},
    )
    candidates = {row.player_fotmob_id: row.player_name for row in result.all()}

    if not candidates:
        return {
            "status": "ok",
            "candidates": 0,
            "fetched": 0,
            "persisted": 0,
        }

    sync_url = _to_sync_url(settings.database_url)
    log.info(
        "[_run_foreign_stats] %d candidate(s), force=%s",
        len(candidates),
        force,
    )
    from scraper.src.player_career_scraper import fetch_and_persist_players

    fetched, persisted = fetch_and_persist_players(candidates, sync_url)
    return {
        "status": "ok",
        "candidates": len(candidates),
        "fetched": fetched,
        "persisted": persisted,
        "unresolved": len(candidates) - fetched,
    }


@router.post("/scrape/quotazioni", summary="Re-import listoni XLSX")
async def trigger_quotazioni(
    quotazioni_dir: str = Query("./quotazioni", description="Directory with XLSX files"),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Re-import listoni and automatically chain neo-arrivo coverage."""
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
        response: dict = {
            "scraper": "quotazioni",
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
        }

        if result.returncode == 0:
            try:
                from ml.data.import_quotations import retry_unmatched
                from sqlalchemy import create_engine
                engine = create_engine(sync_url)
                latest = await db.scalar(
                    sa.text("SELECT MAX(season_start) FROM player_quotations")
                )
                if latest is not None:
                    resolved_df = retry_unmatched(engine, int(latest))
                    response["resolve_unmatched"] = {
                        "status": "ok",
                        "resolved": int(len(resolved_df)) if resolved_df is not None else 0,
                    }
                else:
                    response["resolve_unmatched"] = {
                        "status": "skipped",
                        "reason": "no_quotations",
                    }
            except Exception as exc:
                log.warning(
                    "[trigger_quotazioni] resolve-unmatched failed (non-blocking): %s",
                    exc,
                )
                response["resolve_unmatched"] = {
                    "status": "error",
                    "error": str(exc)[:200],
                }

            try:
                foreign = await _run_foreign_stats(db, force=False)
                response["foreign_stats"] = foreign
            except Exception as exc:
                log.warning(
                    "[trigger_quotazioni] foreign-stats fetch failed (non-blocking): %s",
                    exc,
                )
                response["foreign_stats"] = {
                    "status": "error",
                    "error": str(exc)[:200],
                }

        return ORJSONResponse(response)
    except Exception:
        log.exception("Quotazioni import failed")
        raise HTTPException(status_code=500, detail="Quotazioni import failed. Check server logs.")


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
    """Targeted per-player fallback for MANTRA's neo-arrivo handling."""
    try:
        payload = await _run_foreign_stats(db, force=force)
        if payload.get("status") == "skipped" and payload.get("reason") == "no_quotations":
            raise HTTPException(status_code=400, detail="No quotations imported yet.")
        return ORJSONResponse({"scraper": "foreign-stats", **payload})
    except HTTPException:
        raise
    except Exception:
        log.exception("Foreign career-stats fetch failed")
        raise HTTPException(
            status_code=500,
            detail="Foreign career-stats fetch failed. Check server logs.",
        )


@router.post(
    "/scrape/resolve-unmatched",
    summary="Retry FotMob resolution for unmatched players, then fetch career stats",
)
async def trigger_resolve_unmatched(
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """P2: second-chance ID resolution for match_method='unmatched' rows."""
    try:
        latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
        if latest is None:
            raise HTTPException(status_code=400, detail="No quotations imported yet.")

        sync_url = _to_sync_url(settings.database_url)
        from ml.data.import_quotations import retry_unmatched
        from sqlalchemy import create_engine

        engine = create_engine(sync_url)
        resolved_df = retry_unmatched(engine, int(latest))
        n_resolved = int(len(resolved_df)) if resolved_df is not None else 0

        foreign_payload: dict = {"status": "skipped", "reason": "nothing_resolved"}
        if n_resolved > 0:
            try:
                foreign_payload = await _run_foreign_stats(db, force=False)
            except Exception as exc:
                log.warning(
                    "[resolve-unmatched] foreign-stats after resolve failed (non-blocking): %s",
                    exc,
                )
                foreign_payload = {"status": "error", "error": str(exc)[:200]}

        return ORJSONResponse({
            "scraper": "resolve-unmatched",
            "status": "ok",
            "resolved": n_resolved,
            "foreign_stats": foreign_payload,
        })
    except HTTPException:
        raise
    except Exception:
        log.exception("resolve-unmatched failed")
        raise HTTPException(
            status_code=500,
            detail="resolve-unmatched failed. Check server logs.",
        )


# ── Data Health ──────────────────────────────────────────────────────────────


@router.get("/data-health", summary="Data coverage overview")
async def get_data_health(
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Return coverage status for all data sources."""
    sources: list[dict] = []

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

    mantra_count = await db.scalar(sa.text("SELECT COUNT(*) FROM player_mantra_roles"))
    sources.append({
        "name": "mantra_roles",
        "total_rows": int(mantra_count or 0),
        "status": "ok" if int(mantra_count or 0) > 0 else "missing",
    })

    md_count = await db.scalar(
        sa.text("SELECT COUNT(*) FROM player_matchday_status")
    )
    md_latest = await db.scalar(
        sa.text(
            "SELECT matchday FROM player_matchday_status "
            "WHERE season_start = (SELECT MAX(season_start) FROM player_matchday_status) "
            "ORDER BY matchday DESC LIMIT 1"
        )
    )
    sources.append({
        "name": "matchday_status",
        "total_rows": int(md_count or 0),
        "latest_matchday": md_latest,
        "status": "ok" if int(md_count or 0) > 0 else "missing",
    })

    exp_count = await db.scalar(sa.text("SELECT COUNT(*) FROM expert_ratings"))
    sources.append({
        "name": "expert_ratings",
        "total_rows": int(exp_count or 0),
        "status": "ok" if int(exp_count or 0) > 0 else "missing",
    })

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

    latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
    if latest is not None:
        unmatched_total = await db.scalar(
            sa.text("""
                SELECT COUNT(*) FROM player_id_map
                WHERE season_start = :ss AND match_method = 'unmatched'
                  AND player_fotmob_id IS NULL
            """),
            {"ss": latest},
        )
        resolved_by_retry = await db.scalar(
            sa.text("""
                SELECT COUNT(*) FROM player_id_map
                WHERE season_start = :ss AND match_method = 'fotmob_suggest_retry'
            """),
            {"ss": latest},
        )
        fs_result = await db.execute(
            _FOREIGN_STATS_CANDIDATES_SQL,
            {"season_start": latest, "force": False},
        )
        fs_candidates = len(fs_result.all())
        sources.append({
            "name": "neo_arrivi_coverage",
            "season_start": int(latest),
            "unmatched_total": int(unmatched_total or 0),
            "resolved_by_retry": int(resolved_by_retry or 0),
            "foreign_stats_candidates": fs_candidates,
            "status": (
                "ok"
                if int(unmatched_total or 0) == 0 and fs_candidates == 0
                else "warning"
            ),
        })
    else:
        sources.append({
            "name": "neo_arrivi_coverage",
            "status": "missing",
            "reason": "no_quotations",
        })

    # ── ML coverage ────────────────────────────────────────────────────────
    # The active list is DB-authoritative; ML presence is artifact-authoritative.
    # This keeps the metric directly comparable to a SQL query over
    # player_id_map + results_latest.json and avoids depending on the frontend.
    if latest is not None:
        active_rows = (await db.execute(sa.text("""
            WITH active AS (
                SELECT DISTINCT ON (pq.fantacalcio_id)
                       pq.fantacalcio_id, pim.player_fotmob_id, pq.player_name,
                       pq.team,
                       COALESCE(pss.seasons_in_italy, 0) AS stagioni_it
                FROM player_quotations pq
                LEFT JOIN player_id_map pim
                  ON pim.fantacalcio_id = pq.fantacalcio_id
                 AND pim.season_start = pq.season_start
                LEFT JOIN player_season_aggregates pss
                  ON pss.fantacalcio_id = pim.player_fotmob_id::bigint
                 AND pss.season_start = pq.season_start
                WHERE pq.season_start = :ss
                ORDER BY pq.fantacalcio_id
            )
            SELECT * FROM active ORDER BY player_name
        """), {"ss": latest})).mappings().all()

        from ml.storage.artifact_store import ArtifactStore, R2Config
        store = ArtifactStore(
            local_dir=Path(settings.artifacts_dir),
            r2_config=R2Config(
                endpoint_url=settings.r2_endpoint_url,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_bucket_name,
            ),
        )
        ml_artifact = store.load_json("results_latest.json")
        predictions = (ml_artifact or {}).get("predictions", [])
        ml_ids = {int(p["player_fotmob_id"]) for p in predictions if p.get("player_fotmob_id") is not None}

        n_players = len(active_rows)
        with_ml = sum(1 for r in active_rows if r["player_fotmob_id"] is not None and int(r["player_fotmob_id"]) in ml_ids)
        neo = [r for r in active_rows if int(r["stagioni_it"] or 0) == 0]
        neo_unresolved = [r for r in neo if r["player_fotmob_id"] is None or int(r["player_fotmob_id"]) not in ml_ids]
        coverage = (with_ml / n_players) if n_players else 0.0
        sources.append({
            "name": "ml_coverage",
            "season_start": int(latest),
            "n_players": n_players,
            "n_with_ml_data": with_ml,
            "coverage_pct": round(coverage * 100.0, 1),
            "n_neo_arrivo": len(neo),
            "n_neo_arrivo_unresolved": len(neo_unresolved),
            "neo_arrivo_unresolved": [
                {
                    "fantacalcio_id": int(r["fantacalcio_id"]),
                    "player_name": r["player_name"],
                    "team": r["team"],
                    "player_fotmob_id": int(r["player_fotmob_id"]) if r["player_fotmob_id"] is not None else None,
                }
                for r in neo_unresolved
            ],
            "artifact": "results_latest.json" if ml_artifact is not None else "missing",
            "warning_threshold_pct": round(settings.ml_coverage_warning_threshold * 100.0, 1),
            "status": "ok" if coverage >= settings.ml_coverage_warning_threshold else "warning",
        })
    else:
        sources.append({"name": "ml_coverage", "status": "missing", "reason": "no_quotations"})

    return ORJSONResponse({"sources": sources})


@router.get("/data-health/{source}", summary="Detailed coverage for a source")
async def get_source_health(
    source: str,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Return detailed coverage info for a specific data source."""
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
            "description": "Re-import Fantacalcio listoni XLSX (+ auto neo-arrivi chain)",
            "frequency": "Season start",
            "configurable_params": ["quotazioni_dir"],
        },
        {
            "name": "foreign-stats",
            "description": "Targeted career stats for neo-arrivi missing Serie A history",
            "frequency": "After listone import / on demand",
            "configurable_params": ["force"],
        },
        {
            "name": "resolve-unmatched",
            "description": "Retry FotMob ID resolution for unmatched players",
            "frequency": "After listone import / on demand",
            "configurable_params": [],
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
