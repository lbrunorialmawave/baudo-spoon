"""Routers for ML-driven insights: player predictions and clustering intelligence.

Routes
------
GET  /predictions/players                  — Paginated player predictions (ML + DB metadata).
GET  /predictions/next-season              — Next-season projected ratings.
GET  /predictions/hybrid                   — Paginated hybrid MANTRA+ML predictions.
GET  /predictions/hybrid/stats             — Hybrid aggregate statistics.
GET  /predictions/hybrid/config            — Current hybrid configuration.
PUT  /predictions/hybrid/config            — Update hybrid configuration (admin).
POST /predictions/hybrid/run               — (Re)generate hybrid results.
GET  /predictions/hybrid/preview           — Preview-only hybrid results (admin).
GET  /intelligence/clustering/players      — Full cluster membership list.
GET  /intelligence/clustering/alternatives — Low-cost player clones (requires API key).
POST /intelligence/cache/invalidate        — Evict Redis cache (requires API key).
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ml.storage.artifact_store import ArtifactAvailability, ArtifactStore

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, require_role
from ..config import settings
from ..models import PlayerSeasonStat, Season
from ..schemas import (
    AlternativesResponse,
    ClusteringStatsSchema,
    LowCostAlternativeSchema,
    ModelComparisonSchema,
    NextSeasonPredictionSchema,
    PlayerClusterSchema,
    PlayerPredictionSchema,
    PlayerVarSchema,
    VarResultsResponse,
)

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

# ── Shared dependency ─────────────────────────────────────────────────────────


def get_repository(request: Request) -> DataRepository:
    """Retrieve the application-scoped DataRepository from app.state."""
    repo: DataRepository | None = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="ML data repository not initialised")
    return repo


def get_artifact_store(request: Request) -> ArtifactStore:
    """Retrieve the application-scoped ArtifactStore from app.state.

    Unica porta d'ingresso R2/disco locale — vedi design doc "R2 come
    source of truth per gli artefatti ML/MANTRA" (2026-08-02).
    """
    store: ArtifactStore | None = getattr(request.app.state, "artifact_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="ML artifact store not initialised")
    return store


# ── Predictions router (public) ───────────────────────────────────────────────

predictions_router = APIRouter(prefix="/predictions", tags=["predictions"])


@predictions_router.get(
    "/players",
    response_class=ORJSONResponse,
    summary="Paginated ML player predictions",
    description=(
        "Returns player Fantacalcio rating predictions from the latest ML run, "
        "enriched with team metadata from **PlayerSeasonStat**. "
        "Supports filtering by player name, team, and canonical role."
    ),
    responses={
        200: {"description": "Paginated prediction envelope with run metadata"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_player_predictions(
    player: Optional[str] = Query(None, description="Filter by player name (partial, case-insensitive)"),
    team: Optional[str] = Query(None, description="Filter by team name (partial, case-insensitive)"),
    role: Optional[str] = Query(None, description="Canonical role: GK, DEF, MID, FWD"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    size: int = Query(50, ge=1, le=200, description="Items per page"),
    repo: DataRepository = Depends(get_repository),
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    try:
        raw = await repo.get_predictions()
        meta = await repo.get_run_metadata()
        model_comparison = await repo.get_model_comparison()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Enrich: build player_name → DB metadata lookup from the most-recent season.
    db_result = await db.execute(
        select(
            PlayerSeasonStat.player_name,
            PlayerSeasonStat.player_fotmob_id,
            PlayerSeasonStat.team_name,
        )
        .join(PlayerSeasonStat.season)
        .distinct(PlayerSeasonStat.player_fotmob_id)
        .order_by(PlayerSeasonStat.player_fotmob_id, Season.season_start.desc())
    )
    db_lookup: dict[str, dict] = {
        row.player_name: {"player_fotmob_id": row.player_fotmob_id, "team_name": row.team_name}
        for row in db_result.all()
    }

    # Merge ML records with DB metadata and apply filters.
    items: list[PlayerPredictionSchema] = []
    for r in raw:
        name: str = r.get("player_name", "")
        db_meta = db_lookup.get(name, {})
        item = PlayerPredictionSchema(
            player_name=name,
            player_fotmob_id=r.get("player_fotmob_id") or db_meta.get("player_fotmob_id"),
            team_name=r.get("team_name") or db_meta.get("team_name"),
            canonical_role=r.get("canonical_role"),
            # Artifact uses season_start; older consumers may still emit season.
            season=r.get("season_start") or r.get("season"),
            fantavoto_medio=r.get("fantavoto_medio"),
            # Artifact key is predicted_fantavoto; keep predicted as fallback.
            predicted=float(r.get("predicted_fantavoto") if r.get("predicted_fantavoto") is not None else r.get("predicted", 0.0)),
            confidence=r.get("confidence"),
            prediction_interval_low=r.get("prediction_interval_low"),
            prediction_interval_high=r.get("prediction_interval_high"),
            expected_minutes=r.get("expected_minutes"),
            # PR9 output-reliability fields (absent on older artifacts).
            sample_cohort=r.get("sample_cohort"),
            ml_values_noisy=r.get("ml_values_noisy"),
            predicted_display=(
                r.get("predicted_fantavoto_display")
                if r.get("predicted_fantavoto_display") is not None
                else r.get("predicted_display")
            ),
        )
        if player and player.lower() not in name.lower():
            continue
        if team and (not item.team_name or team.lower() not in item.team_name.lower()):
            continue
        if role and item.canonical_role != role.upper():
            continue
        items.append(item)

    total = len(items)
    page_items = items[(page - 1) * size : page * size]

    payload = {
        "runId": meta["run_id"],
        "bestModel": meta["best_model"],
        "rolePartitioned": meta["role_partitioned"],
        "modelComparison": [ModelComparisonSchema(**m).model_dump(by_alias=True) for m in model_comparison],
        "total": total,
        "page": page,
        "size": size,
        "items": [p.model_dump(by_alias=True) for p in page_items],
    }
    return ORJSONResponse(content=payload)


@predictions_router.get(
    "/next-season",
    response_class=ORJSONResponse,
    summary="Next-season projected player ratings",
    description=(
        "Forward-projected Fantacalcio ratings generated when the pipeline was "
        "run with ``--predict-next``. Returns 404 when not available."
    ),
    responses={
        200: {"description": "List of next-season predictions"},
        404: {"description": "No next-season predictions available"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_next_season_predictions(
    player: Optional[str] = Query(None, description="Filter by player name (partial)"),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        raw = await repo.get_next_season_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not raw:
        raise HTTPException(status_code=404, detail="No next-season predictions in current artifact")

    items = [NextSeasonPredictionSchema(**r) for r in raw]
    if player:
        q = player.lower()
        items = [i for i in items if q in i.player_name.lower()]

    return ORJSONResponse(content=[i.model_dump(by_alias=True) for i in items])


# ── Intelligence router (API-key protected + rate-limited) ────────────────────

intelligence_router = APIRouter(
    prefix="/intelligence",
    tags=["intelligence"],
    dependencies=[Depends(require_role("member")), Depends(rate_limit)],
)


@intelligence_router.get(
    "/clustering/players",
    response_class=ORJSONResponse,
    summary="Player cluster assignments",
    description=(
        "PCA-reduced cluster membership for every player in the latest ML run. "
        "Useful for building player similarity maps and visualisations."
    ),
    responses={
        200: {"description": "Paginated cluster assignments with clustering stats"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_cluster_players(
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    role: Optional[str] = Query(None, description="Canonical role: GK, DEF, MID, FWD"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        raw = await repo.get_player_clusters()
        stats = await repo.get_clustering_stats()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = [PlayerClusterSchema(**r) for r in raw]
    if cluster_id is not None:
        items = [i for i in items if i.cluster_id == cluster_id]
    if role:
        items = [i for i in items if i.canonical_role == role.upper()]

    total = len(items)
    page_items = items[(page - 1) * size : page * size]

    payload = {
        "clusteringStats": ClusteringStatsSchema(**stats).model_dump(by_alias=True),
        "total": total,
        "page": page,
        "size": size,
        "items": [p.model_dump(by_alias=True) for p in page_items],
    }
    return ORJSONResponse(content=payload)


@intelligence_router.get(
    "/clustering/alternatives",
    response_class=ORJSONResponse,
    summary="Low-cost player alternatives",
    description=(
        "For each top-percentile player (above the 80th percentile of Fantacalcio "
        "rating) returns cluster-mates from less prestigious clubs — a.k.a. "
        "'budget clones'. Filter by ``top_player_id`` to focus on a single player. "
        "Ideal for budget-constrained Fantacalcio roster construction."
    ),
    responses={
        200: {"description": "Alternatives + clustering metadata"},
        404: {"description": "No recommendations found for requested player"},
        503: {"description": "ML artifact not yet generated"},
    },
)
async def list_low_cost_alternatives(
    top_player_id: Optional[int] = Query(
        None, description="FotMob player ID — filter recommendations for one top player"
    ),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        recs = await repo.get_low_cost_recommendations(top_player_id=top_player_id)
        stats = await repo.get_clustering_stats()
        clusters = await repo.get_player_clusters()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if top_player_id is not None and not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for player_id={top_player_id}",
        )

    response = AlternativesResponse(
        clustering_stats=ClusteringStatsSchema(**stats),
        player_clusters=[PlayerClusterSchema(**c) for c in clusters],
        low_cost_recommendations=[LowCostAlternativeSchema(**r) for r in recs],
    )
    return ORJSONResponse(content=response.model_dump(by_alias=True))


@intelligence_router.get(
    "/var/players",
    response_class=ORJSONResponse,
    summary="Value Above Replacement for all players",
    description=(
        "Returns VAR and Expected Surplus Value (ESV) per player from the latest ML run. "
        "ESV > 0 means the player is expected to be underpriced at auction. "
        "**Note**: `calibrated=false` until real auction history is fitted to the demand curve."
    ),
    responses={
        200: {"description": "VAR results sorted by ESV descending"},
        503: {"description": "ML artifact not yet generated or VAR not computed"},
    },
)
async def list_var_players(
    role: Optional[str] = Query(None, description="Filter by role: P, D, C, A"),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        meta = await repo.get_run_metadata()
        raw = await repo.get_var_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not raw:
        raise HTTPException(
            status_code=503,
            detail="VAR results not available in current artifact. Re-run the ML pipeline.",
        )

    items = [PlayerVarSchema(**r) for r in raw]
    if role:
        items = [i for i in items if i.role == role.upper()]

    calibrated = all(i.calibrated for i in items)
    response = VarResultsResponse(
        run_id=meta["run_id"],
        calibrated=calibrated,
        total=len(items),
        items=items,
    )
    return ORJSONResponse(content=response.model_dump(by_alias=True))


@intelligence_router.post(
    "/cache/invalidate",
    summary="Invalidate ML result cache",
    description=(
        "Evicts Redis-cached ML artifact entries. "
        "Call this after deploying a new ML pipeline run to ensure stale data is not served."
    ),
    responses={200: {"description": "Cache invalidated successfully"}},
)
async def invalidate_cache(
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    await repo.invalidate_cache()
    return ORJSONResponse(content={"detail": "Cache invalidated"})


# ── Hybrid endpoints (under /predictions) ─────────────────────────────────────


class HybridSortField(str, Enum):
    player_name = "playerName"
    fp_ibrido = "fpIbrido"
    confidence_score = "confidenceScore"
    expected_value = "expectedValue"
    fp_gap = "fpGap"
    predicted_fantavoto = "predictedFantavoto"
    fp_corr = "FP_Corr"
    vr = "VR"


# Transform selected snake_case keys to camelCase for the frontend. Explicit
# mapping ensures FP_Corr, CP_Corr etc. are handled correctly. Module-level
# (not just used by /predictions/hybrid) so /overview/players can reuse the
# exact same conversion without duplicating it.
_CAMEL_OVERRIDES = {
    "player_name": "playerName",
    "player_fotmob_id": "playerFotmobId",
    "season_start": "seasonStart",
    "ruolo_primario": "ruoloPrimario",
    "ruoli_mantra": "ruoliMantra",
    "has_ml_data": "hasMlData",
    "predicted_fantavoto": "predictedFantavoto",
    "prediction_std": "predictionStd",
    "expected_minutes": "expectedMinutes",
    "var_score": "varScore",
    "next_season_predicted": "nextSeasonPredicted",
    "ml_score_norm": "mlScoreNorm",
    "confidence_score": "confidenceScore",
    "fp_gap": "fpGap",
    "fp_ibrido": "fpIbrido",
    "expected_value": "expectedValue",
    "prezzo_massimo": "prezzoMassimo",
    "hybrid_labels": "hybridLabels",
    # ML foreign-league fallback (trainer → merger → scoring → overview drawer)
    "is_foreign_fallback": "isForeignFallback",
    # PR9 output-reliability (trainer → predictions / hybrid / overview)
    "sample_cohort": "sampleCohort",
    "ml_values_noisy": "mlValuesNoisy",
    "predicted_fantavoto_display": "predictedFantavotoDisplay",
    "predicted_display": "predictedDisplay",
    "predicted_next_fantavoto_display": "predictedNextFantavotoDisplay",
    # Mixed-case keys that must pass through unchanged
    "FP_Corr": "FP_Corr",
    "CP_Corr": "CP_Corr",
    "FP_Mantra": "FP_Mantra",
    "Prezzo_Massimo": "prezzoMassimo",
    # Fase7 v2 — two independent axes (Rendimento/Affidabilità, Prezzo/Valore),
    # each with a label, a reason, and a numeric gap for a frontend "stars"
    # confidence indicator. Redundant with the generic uppercase-passthrough
    # rule below, kept explicit for readability.
    "Fase7_Rendimento": "Fase7_Rendimento",
    "Fase7_Rendimento_Motivo": "Fase7_Rendimento_Motivo",
    "Fase7_Rendimento_Gap": "Fase7_Rendimento_Gap",
    "Fase7_Prezzo": "Fase7_Prezzo",
    "Fase7_Prezzo_Motivo": "Fase7_Prezzo_Motivo",
    "Fase7_Prezzo_Gap": "Fase7_Prezzo_Gap",
}


def _to_camel(key: str) -> str:
    if key in _CAMEL_OVERRIDES:
        return _CAMEL_OVERRIDES[key]
    # Keys already camelCase pass through unchanged
    if "_" not in key:
        return key
    # Mixed-case keys with underscore (FP_Corr, CP_Corr, FP_Mantra, Prezzo_Massimo,
    # Fase7_Rendimento, Fase7_Prezzo, ...) contain acronym parts — pass them
    # through as-is
    if any(c.isupper() for c in key):
        return key
    # Generic snake_case → camelCase
    parts = key.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


async def _load_hybrid_results(
    artifact_store: ArtifactStore, db: AsyncSession, repo: DataRepository
) -> tuple[int, dict[str, Any]]:
    """Find the latest available hybrid artefact: local disk → R2 (season
    fallback), via ArtifactStore.

    Season candidates are resolved from the latest imported quotations
    (DB-driven, not a calendar guess) rather than a hardcoded list — see
    DataRepository.resolve_season_candidates.

    Replaces the old ``_find_hybrid_path``, which every reader endpoint
    actually called and which used a bare ``Path.exists()`` with **no**
    R2 fallback at all — a stricter instance of P1/P4 than the (unused)
    ``_load_hybrid_results`` sketch suggested. See design doc "R2 come
    source of truth per gli artefatti ML/MANTRA" (2026-08-02), Fase 4.
    """
    for season in await repo.resolve_season_candidates(db):
        data = await artifact_store.load_json_async(f"mantra_ibrido_results_{season}.json")
        if data is not None:
            return season, data
    raise FileNotFoundError(
        "No hybrid results found. Run POST /predictions/hybrid/run first."
    )


@predictions_router.get(
    "/hybrid",
    response_class=ORJSONResponse,
    summary="Paginated hybrid MANTRA+ML predictions",
    description=(
        "Returns every player scored with both MANTRA pillars and ML predictions, "
        "plus computed hybrid scores (FP_Ibrido, Confidence, Expected Value, etc.) "
        "and classification labels."
    ),
    responses={
        200: {"description": "Paginated hybrid predictions"},
        503: {"description": "Hybrid artefact not available"},
    },
)
async def list_hybrid_predictions(
    ruolo: Optional[str] = Query(None, description="Filter by MANTRA primary role"),
    search: Optional[str] = Query(None, description="Search by player name"),
    confidence_min: Optional[float] = Query(None, ge=0, le=100, alias="confidenceMin"),
    label: Optional[str] = Query(None, description="Filter by hybrid label (e.g. ML_Confirmed)"),
    sort_by: Optional[HybridSortField] = Query(None, alias="sortBy"),
    sort_dir: Optional[str] = Query("asc", alias="sortDir"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=2000),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        _, data = await _load_hybrid_results(artifact_store, db, repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    players = data.get("players", [])

    # Filters
    if ruolo:
        players = [p for p in players if p.get("ruolo_primario") == ruolo]
    if search:
        q = search.lower()
        players = [p for p in players if q in str(p.get("player_name", "")).lower()]
    if confidence_min is not None:
        players = [p for p in players if (p.get("confidenceScore") or 0) >= confidence_min]
    if label:
        players = [p for p in players if label in p.get("hybridLabels", [])]

    # Sorting
    if sort_by:
        reverse = sort_dir == "desc"
        players.sort(
            key=lambda p: (
                p.get(sort_by.value) if p.get(sort_by.value) is not None
                else -999999
            ),
            reverse=reverse,
        )

    total = len(players)
    start = (page - 1) * size
    items = players[start:start + size]

    # camelCase conversion via the module-level _to_camel/_CAMEL_OVERRIDES
    # (shared with /overview/players — see routers/overview.py).
    camel_items = [
        {_to_camel(k): v for k, v in p.items()}
        for p in items
    ]

    return ORJSONResponse({
        "total": total,
        "page": page,
        "size": size,
        "items": camel_items,
        "meta": data.get("meta"),
    })


@predictions_router.get(
    "/hybrid/stats",
    response_class=ORJSONResponse,
    summary="Hybrid aggregate statistics",
)
async def get_hybrid_stats(
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        _, data = await _load_hybrid_results(artifact_store, db, repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    players = data.get("players", [])
    if not players:
        return ORJSONResponse({"totalPlayers": 0})

    n_total = len(players)
    n_with_ml = sum(1 for p in players if p.get("has_ml_data"))

    fp_scores = [p["fpIbrido"] for p in players if p.get("fpIbrido") is not None]
    conf_scores = [p["confidenceScore"] for p in players if p.get("confidenceScore") is not None]
    gaps = [p["fpGap"] for p in players if p.get("fpGap") is not None]

    classifications = data.get("classifications", {})
    classification_counts = {k: len(v) for k, v in classifications.items()}

    return ORJSONResponse({
        "totalPlayers": n_total,
        "pctWithMl": round(n_with_ml / max(n_total, 1), 2),
        "avgFpIbrido": round(sum(fp_scores) / max(len(fp_scores), 1), 2) if fp_scores else None,
        "avgConfidenceScore": round(sum(conf_scores) / max(len(conf_scores), 1), 2) if conf_scores else None,
        "avgFpGap": round(sum(gaps) / max(len(gaps), 1), 2) if gaps else None,
        "classificationCounts": classification_counts,
    })


@predictions_router.get(
    "/hybrid/status",
    response_class=ORJSONResponse,
    summary="Check availability of MANTRA, ML predictions, and hybrid results",
    description=(
        "Returns the readiness status of each computation layer without throwing errors. "
        "Use this to show meaningful messages in the UI instead of crashing."
    ),
)
async def get_hybrid_status(
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    """Readiness derives from the ArtifactStore (Redis-independent: local
    disk → R2), never from ``Path.exists()`` on this specific instance's
    disk. On a stateless multi-instance deployment, an artefact freshly
    produced by another instance (or a prior deploy) and only present on
    R2 must still report as ready here — see design doc "R2 come source
    of truth per gli artefatti ML/MANTRA" (2026-08-02), Fase 3.
    """
    artifacts_dir = Path(settings.artifacts_dir)
    seasons = await repo.resolve_season_candidates(db)

    ml_filename = "results_latest.json"
    mantra_filenames = [f"mantra_results_{season}.json" for season in seasons]
    hybrid_filenames = [f"mantra_ibrido_results_{season}.json" for season in seasons]
    all_filenames = [ml_filename, *mantra_filenames, *hybrid_filenames]

    availabilities = await asyncio.gather(
        *(artifact_store.exists_async(filename) for filename in all_filenames)
    )

    for filename, availability in zip(all_filenames, availabilities):
        if availability is ArtifactAvailability.R2_UNREACHABLE:
            # Non deve mai travestirsi da "dati non ancora pronti" (P4):
            # logghiamo esplicitamente per renderlo osservabile/allarmabile.
            log.warning("hybrid/status: R2 unreachable while checking key=%s", filename)

    ready = {"local", "remote_only"}
    availability_by_filename = dict(zip(all_filenames, availabilities))

    ml_exists = availability_by_filename[ml_filename].value in ready

    mantra_available: list[dict[str, Any]] = [
        {"season": season, "path": str(artifacts_dir / filename)}
        for season, filename in zip(seasons, mantra_filenames)
        if availability_by_filename[filename].value in ready
    ]
    hybrid_available: list[dict[str, Any]] = [
        {"season": season, "path": str(artifacts_dir / filename)}
        for season, filename in zip(seasons, hybrid_filenames)
        if availability_by_filename[filename].value in ready
    ]

    return ORJSONResponse({
        "mlPredictionsReady": ml_exists,
        "mantraResults": mantra_available,
        "hybridResults": hybrid_available,
        "hybridReady": (
            ml_exists
            and len(mantra_available) > 0
            and any(h["season"] in {m["season"] for m in mantra_available} for h in hybrid_available)
        ),
    })


@predictions_router.get(
    "/hybrid/config",
    response_class=ORJSONResponse,
    summary="Current hybrid configuration",
)
async def get_hybrid_config() -> ORJSONResponse:
    from ml.mantra_ibrido.config_store import load_config

    cfg = load_config()
    return ORJSONResponse({
        "PESO_MANTRA": cfg.PESO_MANTRA,
        "PESO_ML": cfg.PESO_ML,
        "W_PREDICTION_STD": cfg.W_PREDICTION_STD,
        "W_MINUTES": cfg.W_MINUTES,
        "EV_SCALE_FACTOR": cfg.EV_SCALE_FACTOR,
        "CONFIDENZA_SOGLIA": cfg.CONFIDENZA_SOGLIA,
        "ML_BOOST_SOGLIA": cfg.ML_BOOST_SOGLIA,
        "ML_BOOST_FP_CORR_MAX": cfg.ML_BOOST_FP_CORR_MAX,
        "ML_TOP_PRED_MIN": cfg.ML_TOP_PRED_MIN,
        "ML_TOP_BOOST_MIN": cfg.ML_TOP_BOOST_MIN,
        "SOGLIA_GAP_ALERT": cfg.SOGLIA_GAP_ALERT,
        "SLEEPER_FP_CORR_MAX": cfg.SLEEPER_FP_CORR_MAX,
        "SLEEPER_ML_NORM_MIN": cfg.SLEEPER_ML_NORM_MIN,
        "BEST_VALUE_VR_MIN": cfg.BEST_VALUE_VR_MIN,
        "BEST_VALUE_FP_IBRIDO_MIN": cfg.BEST_VALUE_FP_IBRIDO_MIN,
        "MINUTES_RISK_MAX": cfg.MINUTES_RISK_MAX,
    })


@predictions_router.put(
    "/hybrid/config",
    response_class=ORJSONResponse,
    summary="Update hybrid configuration (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def update_hybrid_config(
    body: dict[str, Any] = Body(...),
) -> ORJSONResponse:
    from ml.mantra_ibrido.config_store import update_config

    try:
        cfg = update_config(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ORJSONResponse({
        "PESO_MANTRA": cfg.PESO_MANTRA,
        "PESO_ML": cfg.PESO_ML,
        "W_PREDICTION_STD": cfg.W_PREDICTION_STD,
        "W_MINUTES": cfg.W_MINUTES,
        "EV_SCALE_FACTOR": cfg.EV_SCALE_FACTOR,
        "CONFIDENZA_SOGLIA": cfg.CONFIDENZA_SOGLIA,
        "ML_BOOST_SOGLIA": cfg.ML_BOOST_SOGLIA,
        "ML_BOOST_FP_CORR_MAX": cfg.ML_BOOST_FP_CORR_MAX,
        "ML_TOP_PRED_MIN": cfg.ML_TOP_PRED_MIN,
        "ML_TOP_BOOST_MIN": cfg.ML_TOP_BOOST_MIN,
        "SOGLIA_GAP_ALERT": cfg.SOGLIA_GAP_ALERT,
        "SLEEPER_FP_CORR_MAX": cfg.SLEEPER_FP_CORR_MAX,
        "SLEEPER_ML_NORM_MIN": cfg.SLEEPER_ML_NORM_MIN,
        "BEST_VALUE_VR_MIN": cfg.BEST_VALUE_VR_MIN,
        "BEST_VALUE_FP_IBRIDO_MIN": cfg.BEST_VALUE_FP_IBRIDO_MIN,
        "MINUTES_RISK_MAX": cfg.MINUTES_RISK_MAX,
    })


@predictions_router.post(
    "/hybrid/run",
    response_class=ORJSONResponse,
    summary="(Re)generate hybrid results",
    dependencies=[Depends(require_role("member"))],
)
async def run_hybrid(
    season_start: Optional[int] = Query(None, ge=2020, le=2030),
    persist: bool = Query(True, description="If false, writes to preview file only"),
    overrides: Optional[dict[str, Any]] = Body(None),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    from ml.mantra_ibrido.config_store import load_config, update_config
    from ml.mantra_ibrido.runner import run_hybrid_computation

    if season_start is None:
        season_start = await repo.get_current_season_start(db)
        if season_start is None:
            raise HTTPException(status_code=400, detail="No quotations imported yet; pass season_start explicitly.")

    artifacts_dir = Path(settings.artifacts_dir)
    mantra_filename = f"mantra_results_{season_start}.json"
    mantra_path = artifacts_dir / mantra_filename
    ml_path = artifacts_dir / "results_latest.json"

    # ArtifactStore.load_json_async checks local disk first, then R2,
    # downloading into the local cache as a side effect — run_hybrid_computation
    # below reads these as plain Paths, so we need the local cache warm
    # regardless of which instance originally produced the MANTRA run.
    mantra_data = await artifact_store.load_json_async(mantra_filename)
    if mantra_data is None:
        raise HTTPException(
            status_code=503,
            detail=f"MANTRA results not found for season {season_start}. Run POST /mantra/run first.",
        )
    # Best-effort: ML predictions are optional for hybrid scoring, so a miss
    # here isn't fatal — run_hybrid_computation tolerates a missing ml_path.
    await artifact_store.load_json_async("results_latest.json")

    try:
        if persist:
            # Save overrides permanently if provided
            config = update_config(overrides) if overrides else load_config()
            result = run_hybrid_computation(
                mantra_path, ml_path, artifacts_dir,
                config=config,
                output_filename=None,  # production file
            )
        else:
            # Ephemeral: merge overrides in memory, write to preview file only
            base = load_config()
            from dataclasses import asdict, replace
            effective = replace(
                base,
                **{k: v for k, v in (overrides or {}).items()
                   if hasattr(base, k)},
            )
            result = run_hybrid_computation(
                mantra_path, ml_path, artifacts_dir,
                config=effective,
                output_filename=f"mantra_ibrido_preview_{season_start}.json",
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ORJSONResponse({
        "status": "ok",
        "season": season_start,
        "nPlayers": len(result["players"]),
        "generatedAt": result["meta"]["generatedAt"],
        "persisted": persist,
    })


@predictions_router.get(
    "/hybrid/preview",
    response_class=ORJSONResponse,
    summary="Preview-only hybrid results (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def get_hybrid_preview(
    season_start: Optional[int] = Query(None, ge=2020, le=2030, alias="seasonStart"),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    target_season = season_start or await repo.get_current_season_start(db) or 2025
    filename = f"mantra_ibrido_preview_{target_season}.json"

    data = await artifact_store.load_json_async(filename)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No preview available. Run POST /predictions/hybrid/run with persist=false first.",
        )

    return ORJSONResponse(data)
