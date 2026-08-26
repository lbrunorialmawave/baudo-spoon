"""MANTRA scoring router — computed player evaluations and classifications.

Endpoints
---------
GET  /mantra/players                    — List all players with MANTRA scores
GET  /mantra/teams                      — Distinct teams in the current MANTRA season
GET  /mantra/players/{fantacalcio_id}   — Single player detail with pillar breakdown
GET  /mantra/top/{ruolo}                — Top N by FP_Mantra in a role
GET  /mantra/classifications            — All Fase 7/8 classifications
GET  /mantra/classifications/low-cost   — Low Cost players
GET  /mantra/classifications/watchlist  — Watchlist Giovani
POST /mantra/run                        — Execute MANTRA computation (API key)
GET  /mantra/budget                     — Budget overview
GET  /mantra/stats                      — Summary statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ml.storage.artifact_store import ArtifactStore, R2Config

from ..config import settings
from ..deps import get_db, require_role
from ..services.player_enrichment import enrich_with_matchday_status

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mantra",
    tags=["mantra"],
)

# Unica porta d'ingresso R2/disco locale per questo router. Lazy-init:
# costruita al primo utilizzo con le Settings correnti. Tutta la I/O R2
# passa da ArtifactStore (design doc "R2 come source of truth per gli
# artefatti ML/MANTRA", 2026-08-02).
_artifact_store: ArtifactStore | None = None


def _get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        artifacts_dir = Path(settings.artifacts_dir) if hasattr(settings, 'artifacts_dir') else Path("artifacts")
        _artifact_store = ArtifactStore(
            local_dir=artifacts_dir,
            r2_config=R2Config(
                endpoint_url=settings.r2_endpoint_url,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_bucket_name,
            ),
        )
    return _artifact_store


async def _load_mantra_results(db: AsyncSession) -> dict:
    """Load the latest MANTRA results JSON: local disk → R2, via ArtifactStore.

    Season candidates are resolved from the latest imported quotations
    (DB-driven, not a calendar guess — mirrors matchday.py / gruppo_esperti.py).
    """
    store = _get_artifact_store()
    latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
    candidates = [latest - i for i in range(3)] if latest is not None else [2025, 2024, 2023]
    for season in candidates:
        data = store.load_json(f"mantra_results_{season}.json")
        if data is not None:
            return data
    raise FileNotFoundError(
        "No MANTRA results found. Run POST /mantra/run first."
    )


@router.get(
    "/players",
    response_class=ORJSONResponse,
    summary="List all players with MANTRA scores",
    description="Returns paginated player list with all pillar scores, FP, VR, Prezzo, and Fase 7 label.",
)
async def list_mantra_players(
    ruolo: Optional[str] = Query(None, description="Filter by MANTRA primary role"),
    fase7_rendimento: Optional[str] = Query(
        None, description="Filter by Fase 7 Rendimento/Affidabilità axis label (TOP/CERTEZZA/SCOMMESSA)"
    ),
    fase7_prezzo: Optional[str] = Query(
        None, description="Filter by Fase 7 Prezzo/Valore axis label (AFFARE/GIUSTO/SOPRAVALUTATO)"
    ),
    team: Optional[str] = Query(None, description="Filter by team name"),
    search: Optional[str] = Query(None, description="Search by player name"),
    min_fp: Optional[float] = Query(None, ge=0, le=100, description="Minimum FP_Mantra"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum Prezzo_Massimo"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum Prezzo_Massimo"),
    fantacalcio_ids: Optional[str] = Query(
        None, description="Comma-separated list of fantacalcio_id to include (applied before pagination)"
    ),
    sort_by: Optional[str] = Query(None, description="Sort column (player_name, team, FP_Mantra, VR, Prezzo_Massimo, ruolo_primario)"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction: asc or desc"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = data.get("players", [])

    # Arricchisci ogni player con la titolarità REALE scrapata (probabili
    # formazioni) — status (starter/bench/doubtful) + probability — letta da
    # player_matchday_status per la stagione corrente e il matchday più
    # recente. Chiave di join: fantacalcio_id (coerente tra le due fonti).
    # Sorgente separata da start_probability (ML): non la sovrascriviamo.
    try:
        players = await enrich_with_matchday_status(db, players)
    except Exception:
        # L'arricchimento non deve mai far fallire la lista MANTRA.
        log.warning("matchday_status enrichment skipped", exc_info=True)

    # Apply filters
    if ruolo:
        players = [p for p in players if p.get("ruolo_primario") == ruolo]
    if fase7_rendimento:
        players = [p for p in players if p.get("Fase7_Rendimento") == fase7_rendimento]
    if fase7_prezzo:
        players = [p for p in players if p.get("Fase7_Prezzo") == fase7_prezzo]
    if team:
        players = [p for p in players if team.lower() in p.get("team", "").lower()]
    if search:
        players = [p for p in players if search.lower() in p.get("player_name", "").lower()]
    if min_fp is not None:
        players = [p for p in players if (p.get("FP_Mantra") or 0) >= min_fp]
    if fantacalcio_ids is not None:
        ids = {int(x) for x in fantacalcio_ids.split(",") if x.strip().isdigit()}
        players = [p for p in players if p.get("fantacalcio_id") in ids]

    if min_price is not None:
        players = [p for p in players if (p.get("Prezzo_Massimo") or 0) >= min_price]
    if max_price is not None:
        players = [p for p in players if (p.get("Prezzo_Massimo") or 999) <= max_price]

    # Sorting
    if sort_by:
        reverse = sort_dir == "desc"
        try:
            if sort_by == "ruolo_primario":
                role_order = {r: i for i, r in enumerate(
                    ["Por", "Dc", "Dd", "Ds", "B", "E", "M", "C", "T", "W", "A", "Pc"]
                )}
                players.sort(key=lambda p: role_order.get(p.get("ruolo_primario", ""), 999), reverse=reverse)
            else:
                players.sort(key=lambda p: (p.get(sort_by) if p.get(sort_by) is not None else (
                    "" if sort_by in ("player_name", "team") else -999999
                )), reverse=reverse)
        except Exception:
            log.warning("Sort failed for sort_by=%r — skipping", sort_by)

    total = len(players)
    start = (page - 1) * size
    items = players[start:start + size]

    return ORJSONResponse({
        "total": total,
        "page": page,
        "size": size,
        "items": items,
        "meta": data.get("meta"),
    })


@router.get(
    "/teams",
    response_class=ORJSONResponse,
    summary="Distinct teams present in the current MANTRA season",
    description="Teams derived from the same resolved season as GET /mantra/players, "
                "so the list never includes a team with zero players (or omits one that has some).",
)
async def list_mantra_teams(db: AsyncSession = Depends(get_db)) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    teams = sorted({p["team"] for p in data.get("players", []) if p.get("team")})
    return ORJSONResponse({"teams": teams})


@router.get(
    "/players/{fantacalcio_id}",
    response_class=ORJSONResponse,
    summary="Single player detail with pillar breakdown",
)
async def get_mantra_player(
    fantacalcio_id: int,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    for p in data.get("players", []):
        if p.get("fantacalcio_id") == fantacalcio_id:
            return ORJSONResponse({
                "player": p,
                "classifications": {
                    k: v for k, v in data.get("classifications", {}).items()
                    if any(
                        name == p["player_name"]
                        for name in (v if isinstance(v, list) else [])
                    )
                },
            })

    raise HTTPException(status_code=404, detail=f"Player {fantacalcio_id} not found")


@router.get(
    "/top/{ruolo}",
    response_class=ORJSONResponse,
    summary="Top N by FP_Mantra in a role",
)
async def top_per_ruolo(
    ruolo: str,
    limit: int = Query(15, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = [
        p for p in data.get("players", [])
        if p.get("ruolo_primario") == ruolo
    ]
    players.sort(key=lambda p: p.get("FP_Mantra", 0), reverse=True)

    return ORJSONResponse({
        "ruolo": ruolo,
        "limit": limit,
        "items": players[:limit],
    })


@router.get(
    "/classifications",
    response_class=ORJSONResponse,
    summary="All Fase 7/8 classifications",
)
async def get_classifications(db: AsyncSession = Depends(get_db)) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ORJSONResponse(data.get("classifications", {}))


@router.get(
    "/classifications/low-cost",
    response_class=ORJSONResponse,
    summary="Low Cost players",
)
async def get_low_cost(
    titolari_only: bool = Query(False, description="Only Low Cost Titolari"),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    key = "low_cost_titolari" if titolari_only else "low_cost"
    items = data.get("classifications", {}).get(key, [])
    # Return full player objects for low-cost
    player_names = set(items)
    players = [p for p in data.get("players", []) if p.get("player_name") in player_names]
    players.sort(key=lambda p: p.get("VR", 0), reverse=True)

    return ORJSONResponse({"category": key, "count": len(players), "items": players})


@router.get(
    "/classifications/watchlist",
    response_class=ORJSONResponse,
    summary="Watchlist Giovani",
)
async def get_watchlist(db: AsyncSession = Depends(get_db)) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    items = data.get("classifications", {}).get("watchlist_giovani", [])
    player_names = set(items)
    players = [p for p in data.get("players", []) if p.get("player_name") in player_names]

    return ORJSONResponse({"count": len(players), "items": players})


@router.post(
    "/run",
    response_class=ORJSONResponse,
    summary="Execute MANTRA computation (admin required)",
    dependencies=[Depends(require_role("admin"))],
)
async def run_mantra(
    season_start: Optional[int] = Query(None, ge=2020, le=2030),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Trigger the MANTRA scoring pipeline for a given season.

    Requires API key. Runs synchronously (may take several seconds).
    Results are saved to the artifacts directory and served by GET endpoints.
    If ``season_start`` is omitted, resolves to the latest season present in
    ``player_quotations`` (DB-driven, not a calendar guess).
    """
    if season_start is None:
        latest = await db.scalar(sa.text("SELECT MAX(season_start) FROM player_quotations"))
        if latest is None:
            raise HTTPException(status_code=400, detail="No quotations imported yet; pass season_start explicitly.")
        season_start = latest

    try:
        from sqlalchemy import create_engine
        from ml.mantra.runner import run_mantra as compute

        # Build sync engine URL from settings (avoids URL encoding issues in bind)
        sync_url = settings.database_url
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        sync_url = sync_url.replace("postgres+asyncpg://", "postgres+psycopg2://")
        sync_url = sync_url.replace("?ssl=", "?sslmode=").replace("&ssl=", "&sslmode=")
        sync_engine = create_engine(sync_url)

        artifacts_dir = Path(settings.artifacts_dir) if hasattr(settings, 'artifacts_dir') else Path("artifacts")
        result = compute(sync_engine, season_start=season_start, output_dir=artifacts_dir)

        return ORJSONResponse({
            "status": "ok",
            "season_start": season_start,
            "n_players": result["meta"]["n_players"],
            "generated_at": result["meta"]["generated_at"],
        })
    except Exception as e:
        log.exception("MANTRA computation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/budget",
    response_class=ORJSONResponse,
    summary="Budget overview",
)
async def get_budget_overview(db: AsyncSession = Depends(get_db)) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = data.get("players", [])
    total_budget = 500

    # Calculate spending by role group
    role_groups = {
        "Portieri (Por)": ["Por"],
        "Difesa Pura (Dc,B,Dd,Ds)": ["Dc", "B", "Dd", "Ds"],
        "Ibridi (E,M)": ["E", "M"],
        "Centro (C)": ["C"],
        "Fantasia (T,W)": ["T", "W"],
        "Attacco (A,Pc)": ["A", "Pc"],
    }

    budget_by_group = {}
    for group_name, roles in role_groups.items():
        group_players = [p for p in players if p.get("ruolo_primario") in roles]
        avg_price = sum(p.get("Prezzo_Massimo", 0) for p in group_players) / max(len(group_players), 1)
        budget_by_group[group_name] = {
            "count": len(group_players),
            "avg_prezzo_massimo": round(avg_price, 2),
            "estimated_cost": round(avg_price * 3, 2),  # rough estimate for 3 players
        }

    return ORJSONResponse({
        "budget_totale": total_budget,
        "by_role_group": budget_by_group,
    })


@router.get(
    "/stats",
    response_class=ORJSONResponse,
    summary="Summary statistics",
)
async def get_mantra_stats(db: AsyncSession = Depends(get_db)) -> ORJSONResponse:
    try:
        data = await _load_mantra_results(db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = data.get("players", [])

    if not players:
        return ORJSONResponse({"count": 0})

    fase7_rendimento_counts: dict[str, int] = {}
    fase7_prezzo_counts: dict[str, int] = {}
    for p in players:
        rend_label = p.get("Fase7_Rendimento") or "none"
        fase7_rendimento_counts[rend_label] = fase7_rendimento_counts.get(rend_label, 0) + 1
        prezzo_label = p.get("Fase7_Prezzo") or "none"
        fase7_prezzo_counts[prezzo_label] = fase7_prezzo_counts.get(prezzo_label, 0) + 1

    avg_fp = sum(p.get("FP_Mantra", 0) for p in players) / len(players)
    avg_vr = sum(p.get("VR", 0) for p in players) / len(players)

    return ORJSONResponse({
        "total_players": len(players),
        "season_start": data.get("meta", {}).get("season_start"),
        "avg_fp_mantra": round(avg_fp, 2),
        "avg_vr": round(avg_vr, 2),
        "fase7_rendimento_distribution": fase7_rendimento_counts,
        "fase7_prezzo_distribution": fase7_prezzo_counts,
    })
