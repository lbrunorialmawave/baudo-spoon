"""MANTRA scoring router — computed player evaluations and classifications.

Endpoints
---------
GET  /mantra/players                    — List all players with MANTRA scores
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

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..deps import get_db, verify_api_key

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mantra",
    tags=["mantra"],
)


def _load_mantra_results() -> dict:
    """Load the latest MANTRA results JSON from the artifacts directory."""
    artifacts_dir = Path(settings.artifacts_dir) if hasattr(settings, 'artifacts_dir') else Path("artifacts")
    # Try most recent season first
    for season in [2026, 2025, 2024]:
        path = artifacts_dir / f"mantra_results_{season}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
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
    fase7: Optional[str] = Query(None, description="Filter by Fase 7 label (TOP/AFFARE/...)"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    search: Optional[str] = Query(None, description="Search by player name"),
    min_fp: Optional[float] = Query(None, ge=0, le=100, description="Minimum FP_Mantra"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum Prezzo_Massimo"),
    sort_by: Optional[str] = Query(None, description="Sort column (player_name, team, FP_Mantra, VR, Prezzo_Massimo, ruolo_primario)"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction: asc or desc"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
) -> ORJSONResponse:
    try:
        data = _load_mantra_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = data.get("players", [])

    # Apply filters
    if ruolo:
        players = [p for p in players if p.get("ruolo_primario") == ruolo]
    if fase7:
        players = [p for p in players if p.get("Fase7") == fase7]
    if team:
        players = [p for p in players if team.lower() in p.get("team", "").lower()]
    if search:
        players = [p for p in players if search.lower() in p.get("player_name", "").lower()]
    if min_fp is not None:
        players = [p for p in players if (p.get("FP_Mantra") or 0) >= min_fp]
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
            pass  # fallback: no sort on unknown column

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
    "/players/{fantacalcio_id}",
    response_class=ORJSONResponse,
    summary="Single player detail with pillar breakdown",
)
async def get_mantra_player(fantacalcio_id: int) -> ORJSONResponse:
    try:
        data = _load_mantra_results()
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
) -> ORJSONResponse:
    try:
        data = _load_mantra_results()
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
async def get_classifications() -> ORJSONResponse:
    try:
        data = _load_mantra_results()
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
) -> ORJSONResponse:
    try:
        data = _load_mantra_results()
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
async def get_watchlist() -> ORJSONResponse:
    try:
        data = _load_mantra_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    items = data.get("classifications", {}).get("watchlist_giovani", [])
    player_names = set(items)
    players = [p for p in data.get("players", []) if p.get("player_name") in player_names]

    return ORJSONResponse({"count": len(players), "items": players})


@router.post(
    "/run",
    response_class=ORJSONResponse,
    summary="Execute MANTRA computation (API key required)",
    dependencies=[Depends(verify_api_key)],
)
async def run_mantra(
    season_start: int = Query(2025, ge=2020, le=2030),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    """Trigger the MANTRA scoring pipeline for a given season.

    Requires API key. Runs synchronously (may take several seconds).
    Results are saved to the artifacts directory and served by GET endpoints.
    """
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
async def get_budget_overview() -> ORJSONResponse:
    try:
        data = _load_mantra_results()
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
async def get_mantra_stats() -> ORJSONResponse:
    try:
        data = _load_mantra_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    players = data.get("players", [])

    if not players:
        return ORJSONResponse({"count": 0})

    fase7_counts: dict[str, int] = {}
    for p in players:
        label = p.get("Fase7") or "none"
        fase7_counts[label] = fase7_counts.get(label, 0) + 1

    avg_fp = sum(p.get("FP_Mantra", 0) for p in players) / len(players)
    avg_vr = sum(p.get("VR", 0) for p in players) / len(players)

    return ORJSONResponse({
        "total_players": len(players),
        "season_start": data.get("meta", {}).get("season_start"),
        "avg_fp_mantra": round(avg_fp, 2),
        "avg_vr": round(avg_vr, 2),
        "fase7_distribution": fase7_counts,
    })
