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

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ml.storage.artifact_store import ArtifactStore, R2Config

from ..config import settings
from ..deps import get_db, require_role

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mantra",
    tags=["mantra"],
)

# Macro-gruppi di ruolo per la stima d'asta on-demand — stessa suddivisione
# usata in get_budget_overview, ma con slug stabili (contratto API) invece
# delle etichette leggibili usate solo per la UI di quell'endpoint.
RUOLO_MACRO_GRUPPO: dict[str, str] = {
    "Por": "portieri",
    "Dc": "difesa", "B": "difesa", "Dd": "difesa", "Ds": "difesa",
    "E": "ibridi", "M": "ibridi",
    "C": "centro",
    "T": "fantasia", "W": "fantasia",
    "A": "attacco", "Pc": "attacco",
}
RUOLO_MACRO_GRUPPI_VALIDI = frozenset(RUOLO_MACRO_GRUPPO.values())

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
    fase7: Optional[str] = Query(None, description="Filter by Fase 7 label (TOP/AFFARE/...)"),
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
    stima_asta: bool = Query(
        False,
        description="Applica una stima del prezzo d'asta (percentile ruolo + partecipanti) invece della sola quotazione ufficiale",
    ),
    num_partecipanti: Optional[int] = Query(
        None, ge=1, description="Numero partecipanti alla lega (richiesto se stima_asta=True)"
    ),
    percentile_soglia: float = Query(0.7, ge=0.0, le=1.0),
    tasso_base: float = Query(0.05, ge=0.0),
    partecipanti_baseline: int = Query(8, ge=1),
    moltiplicatore_max: float = Query(1.6, ge=1.0),
    override_ruolo_json: Optional[str] = Query(
        None,
        description=(
            "JSON con override per macro-gruppo di ruolo, es. "
            '{"attacco": {"moltiplicatore_max": 2.0, "tasso_base": 0.08, "percentile_soglia": 0.5}}. '
            "Gruppi validi: portieri, difesa, ibridi, centro, fantasia, attacco. "
            "partecipanti_baseline resta sempre globale."
        ),
    ),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    if stima_asta and num_partecipanti is None:
        raise HTTPException(
            status_code=400,
            detail="num_partecipanti è richiesto quando stima_asta=True",
        )

    ruolo_overrides: dict[str, dict[str, float]] = {}
    if override_ruolo_json:
        try:
            ruolo_overrides = json.loads(override_ruolo_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="override_ruolo_json non è un JSON valido")
        if not isinstance(ruolo_overrides, dict):
            raise HTTPException(status_code=400, detail="override_ruolo_json deve essere un oggetto JSON")
        gruppi_sconosciuti = set(ruolo_overrides) - RUOLO_MACRO_GRUPPI_VALIDI
        if gruppi_sconosciuti:
            raise HTTPException(
                status_code=400,
                detail=f"Gruppi di ruolo sconosciuti in override_ruolo_json: {sorted(gruppi_sconosciuti)}",
            )

    try:
        data = await _load_mantra_results(db)
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
    if fantacalcio_ids is not None:
        ids = {int(x) for x in fantacalcio_ids.split(",") if x.strip().isdigit()}
        players = [p for p in players if p.get("fantacalcio_id") in ids]

    # Stima d'asta on-demand: sovrascrive Prezzo_Massimo su copie dei dict
    # (mai in place — `players` condivide le entry con la cache di
    # `_load_mantra_results`) usando lo stesso modello percentile +
    # partecipanti già usato da ml.optimizer/ml.auction. Applicata prima del
    # filtro min_price/max_price così quest'ultimo lavora sul valore stimato.
    if stima_asta:
        from ml.optimizer.inflation import InflationConfig, inflation_multiplier

        # Una InflationConfig per gruppo di ruolo (al più 6), costruita pigramente
        # e riusata per tutti i giocatori dello stesso gruppo. partecipanti_baseline
        # non è overridabile per gruppo: la competizione dipende dalla lega intera,
        # non dal singolo ruolo.
        cfg_per_gruppo: dict[str | None, InflationConfig] = {}

        def _cfg_per_ruolo(ruolo_primario: str | None) -> InflationConfig:
            gruppo = RUOLO_MACRO_GRUPPO.get(ruolo_primario or "")
            if gruppo not in cfg_per_gruppo:
                ov = ruolo_overrides.get(gruppo, {}) if gruppo else {}
                try:
                    cfg_per_gruppo[gruppo] = InflationConfig(
                        inflation_percentile_threshold=ov.get("percentile_soglia", percentile_soglia),
                        base_inflation_rate=ov.get("tasso_base", tasso_base),
                        baseline_participants=partecipanti_baseline,
                        max_inflation_multiplier=ov.get("moltiplicatore_max", moltiplicatore_max),
                    )
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"override_ruolo_json non valido per {gruppo!r}: {e}")
            return cfg_per_gruppo[gruppo]

        stimati = []
        for p in players:
            pct = p.get("Percentile_Ruolo") or 0.0
            base = p.get("Prezzo_Massimo") or 1
            mult = inflation_multiplier(pct, num_partecipanti, _cfg_per_ruolo(p.get("ruolo_primario")))
            p2 = dict(p)
            p2["Prezzo_Base_Listino"] = base
            p2["Prezzo_Massimo"] = round(max(base * mult, 1.0), 2)
            stimati.append(p2)
        players = stimati

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
        "stima_asta_attiva": stima_asta,
        "meta": data.get("meta"),
    })


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
