"""Overview router — unified player view aggregating MANTRA, Hybrid ML,
Gruppo Esperti, and titolarità (real + ML + expert) in one endpoint.

Endpoints
---------
GET  /overview/players  — Paginated aggregated player list.

Reuses the same artifact loading (``_load_hybrid_results``) and camelCase
conversion (``_to_camel``) as ``GET /predictions/hybrid`` (see
``routers/intelligence.py``), and the same enrichment helpers used by
``GET /mantra/players`` (see ``services/player_enrichment.py``) — this
router only combines them, it does not reimplement any of them.
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ml.storage.artifact_store import ArtifactStore

from ..data_repository import DataRepository
from ..deps import get_db
from ..services.player_enrichment import enrich_with_expert_ratings, enrich_with_matchday_status
from .intelligence import _load_hybrid_results, _to_camel, get_artifact_store, get_repository

log = logging.getLogger(__name__)

router = APIRouter(prefix="/overview", tags=["overview"])


class OverviewSortField(str, Enum):
    """Values are the RAW pre-camelCase artifact keys, not the camelCase
    keys the response eventually exposes — sorting runs before `_to_camel`.
    Verified against the actual field-writing code (not assumed):
    - scoring.py writes fpIbrido/confidenceScore/fpGap/expectedValue/mlBoost
      directly in camelCase, so those raw keys ARE camelCase.
    - merger.py writes predicted_fantavoto/prediction_std/... in snake_case,
      and the base MANTRA fields (player_name, ruolo_primario, Pz1, Fase7,
      FP_Mantra, VR) keep the mantra runner's own casing.
    (Note: HybridSortField in intelligence.py sorts `predictedFantavoto` and
    `playerName` against these same raw dicts — since those keys are
    actually snake_case, that lookup silently no-ops on /predictions/hybrid
    today. Pre-existing behavior there, left untouched; fixed here instead
    of copied, since this is a separate enum/endpoint with its own contract.)
    """
    player_name = "player_name"
    team = "team"
    ruolo_primario = "ruolo_primario"
    fp_ibrido = "fpIbrido"
    fp_mantra = "FP_Mantra"
    predicted_fantavoto = "predicted_fantavoto"
    confidence_score = "confidenceScore"
    expected_value = "expectedValue"
    fp_gap = "fpGap"
    fp_corr = "FP_Corr"
    vr = "VR"
    pz1 = "Pz1"
    fase7 = "Fase7"
    probability_scraped = "probability_scraped"
    start_probability = "start_probability"
    expert_totale = "expert_totale"


# Canonical MANTRA role order (12 roles) — same list used by
# GET /mantra/players for sort_by=ruolo_primario (routers/mantra.py).
_ROLE_ORDER = {r: i for i, r in enumerate(
    ["Por", "Dc", "Dd", "Ds", "B", "E", "M", "C", "T", "W", "A", "Pc"]
)}

# Max combinable sort criteria — enough for real use (e.g. "price, then
# expert score, then a tie-break role") without the priority indicator in
# the UI (superscript ¹²³) running out of single-digit ranks.
MAX_SORT_KEYS = 3


def _parse_sort_keys(raw: str | None) -> list[tuple[OverviewSortField, bool]]:
    """Parse a DRF-style combined sort param: 'Pz1,-expert_totale' → a list
    of (field, reverse) in the user's priority order (first = most
    important). '-' prefix means descending.

    Raises 422 (not a silent skip) on an unknown field, a duplicate field,
    or more than MAX_SORT_KEYS criteria — with up to 3 fields combined a
    silently-dropped one would be hard to notice in QA.
    """
    if not raw:
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if len(tokens) > MAX_SORT_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many sort keys ({len(tokens)}); max {MAX_SORT_KEYS}",
        )
    parsed: list[tuple[OverviewSortField, bool]] = []
    seen: set[str] = set()
    for tok in tokens:
        reverse = tok.startswith("-")
        field_raw = tok[1:] if reverse else tok
        try:
            field = OverviewSortField(field_raw)  # lookup by VALUE, not member name
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown sort field: {field_raw!r}")
        if field_raw in seen:
            raise HTTPException(status_code=422, detail=f"Duplicate sort field: {field_raw!r}")
        seen.add(field_raw)
        parsed.append((field, reverse))
    return parsed


def _apply_single_sort(players: list[dict], field: OverviewSortField, reverse: bool) -> None:
    """Apply ONE sort criterion in place, stably. Call this once per
    criterion, in order from least to most important — Python's Timsort is
    stable, so each later call becomes the dominant key while earlier calls
    survive as the tie-break within groups of equal values (this is exactly
    how the pre-existing two-pass string-field branch already works
    internally; composing multiple calls externally follows the same
    principle)."""
    if field is OverviewSortField.ruolo_primario:
        players.sort(
            key=lambda p: _ROLE_ORDER.get(p.get("ruolo_primario", ""), 999),
            reverse=reverse,
        )
    elif field in (OverviewSortField.player_name, OverviewSortField.team, OverviewSortField.fase7):
        # String fields: sort by value in the requested direction, then a
        # second *stable* pass pushes missing values to the end either way
        # (a single reversed tuple key would flip None to the front on
        # desc, which reads as broken, not as "descending").
        players.sort(key=lambda p: p.get(field.value) or "", reverse=reverse)
        players.sort(key=lambda p: p.get(field.value) is None)
    else:
        players.sort(
            key=lambda p: (
                p.get(field.value) if p.get(field.value) is not None else -999999
            ),
            reverse=reverse,
        )


@router.get(
    "/players",
    response_class=ORJSONResponse,
    summary="Paginated aggregated player overview",
    description=(
        "Merges MANTRA pillars, Hybrid ML predictions, real scraped titolarità "
        "(probabili formazioni), and Gruppo Esperti ratings into one row per "
        "player, keyed by fantacalcio_id. Server-side filter/sort/pagination "
        "— no whole-artifact client-side loading."
    ),
    responses={
        200: {"description": "Paginated overview envelope"},
        503: {"description": "Hybrid artefact not available"},
    },
)
async def list_overview_players(
    ruolo: str | None = Query(None, description="Filter by MANTRA primary role"),
    team: str | None = Query(None, description="Filter by team name"),
    search: str | None = Query(None, description="Search by player name"),
    fase7: str | None = Query(None, description="Filter by Fase 7 label (TOP/AFFARE/...)"),
    labels: str | None = Query(
        None, description="Comma-separated hybrid labels (e.g. ML_Confirmed,ML_Top) — OR-matched"
    ),
    confidence_min: float | None = Query(None, ge=0, le=100, alias="confidenceMin"),
    min_fp: float | None = Query(None, ge=0, le=100, description="Minimum FP_Mantra"),
    max_fp: float | None = Query(None, ge=0, le=100, description="Maximum FP_Mantra"),
    min_fp_ibrido: float | None = Query(None, description="Minimum FP Ibrido"),
    max_fp_ibrido: float | None = Query(None, description="Maximum FP Ibrido"),
    min_vr: float | None = Query(None, description="Minimum VR (Valore Reale)"),
    max_vr: float | None = Query(None, description="Maximum VR (Valore Reale)"),
    min_price: float | None = Query(None, ge=0, description="Minimum Pz1 (quotazione ufficiale da listone)"),
    max_price: float | None = Query(None, ge=0, description="Maximum Pz1 (quotazione ufficiale da listone)"),
    fantacalcio_ids: str | None = Query(
        None, description="Comma-separated list of fantacalcio_id to include (applied before pagination)"
    ),
    status_scraped: str | None = Query(
        None, description="Filter by real scraped titolarità (starter/bench/injured/suspended/doubtful)"
    ),
    start_probability_min: float | None = Query(
        None, ge=0, le=100, description="Minimum ML titolarità probability, as a 0-100 percentage"
    ),
    probability_scraped_min: float | None = Query(
        None, ge=0, le=100, description="Minimum real scraped titolarità probability (0-100)"
    ),
    expert_totale_min: float | None = Query(None, ge=0, le=50, description="Minimum Gruppo Esperti totale (/50)"),
    expert_totale_max: float | None = Query(None, ge=0, le=50, description="Maximum Gruppo Esperti totale (/50)"),
    expert_rating_min: float | None = Query(None, ge=0, le=5, description="Minimum Gruppo Esperti rating (1-5 stelle)"),
    expert_titolarita_min: float | None = Query(None, ge=0, le=10, description="Minimum Gruppo Esperti titolarità (1-10)"),
    expert_media_voto_min: float | None = Query(None, ge=0, le=10, description="Minimum Gruppo Esperti media voto (1-10)"),
    expert_salute_min: float | None = Query(None, ge=0, le=10, description="Minimum Gruppo Esperti salute (1-10)"),
    has_ml_data: bool | None = Query(None, description="When true, only players with an ML prediction"),
    has_risk_flag: bool | None = Query(
        None, description="When true, only players with a contextual risk flag (e.g. cambio squadra)"
    ),
    sort_by: str | None = Query(
        None,
        description=(
            "Comma-separated sort keys in priority order, e.g. "
            "'Pz1,-expert_totale' (prefix '-' for descending). "
            f"Up to {MAX_SORT_KEYS} combined."
        ),
    ),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    db: AsyncSession = Depends(get_db),
    repo: DataRepository = Depends(get_repository),
) -> ORJSONResponse:
    try:
        season, data = await _load_hybrid_results(artifact_store, db, repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    players = data.get("players", [])

    # Arricchimenti — mai bloccanti: se una fonte non è disponibile, la lista
    # overview resta comunque servibile con quei campi a None (stesso
    # comportamento fail-safe di GET /mantra/players).
    try:
        players = await enrich_with_matchday_status(db, players)
    except Exception:
        log.warning("matchday_status enrichment skipped", exc_info=True)

    try:
        players = await enrich_with_expert_ratings(db, players, season)
    except Exception:
        log.warning("expert_ratings enrichment skipped", exc_info=True)

    # Filters
    if ruolo:
        players = [p for p in players if p.get("ruolo_primario") == ruolo]
    if team:
        # Exact match, not substring: `team` is always one value picked from
        # GET /mantra/teams' exact list on the frontend, never free text.
        players = [p for p in players if (p.get("team") or "").lower() == team.lower()]
    if search:
        q = search.lower()
        players = [p for p in players if q in str(p.get("player_name", "")).lower()]
    if fase7:
        players = [p for p in players if p.get("Fase7") == fase7]
    if labels:
        wanted = {v.strip() for v in labels.split(",") if v.strip()}
        players = [p for p in players if wanted & set(p.get("hybridLabels") or [])]
    if confidence_min is not None:
        players = [p for p in players if (p.get("confidenceScore") or 0) >= confidence_min]
    if min_fp is not None:
        players = [p for p in players if (p.get("FP_Mantra") or 0) >= min_fp]
    if max_fp is not None:
        players = [p for p in players if (p.get("FP_Mantra") if p.get("FP_Mantra") is not None else 999) <= max_fp]
    if min_fp_ibrido is not None:
        players = [p for p in players if (p.get("fpIbrido") or 0) >= min_fp_ibrido]
    if max_fp_ibrido is not None:
        players = [p for p in players if (p.get("fpIbrido") if p.get("fpIbrido") is not None else 999) <= max_fp_ibrido]
    if min_vr is not None:
        players = [p for p in players if (p.get("VR") or 0) >= min_vr]
    if max_vr is not None:
        players = [p for p in players if (p.get("VR") if p.get("VR") is not None else 999999) <= max_vr]
    if min_price is not None:
        players = [p for p in players if (p.get("Pz1") or 0) >= min_price]
    if max_price is not None:
        players = [p for p in players if (p.get("Pz1") or 999) <= max_price]
    if fantacalcio_ids is not None:
        ids = {int(x) for x in fantacalcio_ids.split(",") if x.strip().isdigit()}
        players = [p for p in players if p.get("fantacalcio_id") in ids]
    if status_scraped:
        players = [p for p in players if p.get("status_scraped") == status_scraped]
    if start_probability_min is not None:
        threshold = start_probability_min / 100.0
        players = [p for p in players if (p.get("start_probability") or 0) >= threshold]
    if probability_scraped_min is not None:
        players = [p for p in players if (p.get("probability_scraped") or 0) >= probability_scraped_min]
    if expert_totale_min is not None:
        players = [p for p in players if (p.get("expert_totale") or 0) >= expert_totale_min]
    if expert_totale_max is not None:
        players = [p for p in players if (p.get("expert_totale") if p.get("expert_totale") is not None else 999) <= expert_totale_max]
    if expert_rating_min is not None:
        players = [p for p in players if (p.get("expert_rating") or 0) >= expert_rating_min]
    if expert_titolarita_min is not None:
        players = [p for p in players if (p.get("expert_titolarita") or 0) >= expert_titolarita_min]
    if expert_media_voto_min is not None:
        players = [p for p in players if (p.get("expert_media_voto") or 0) >= expert_media_voto_min]
    if expert_salute_min is not None:
        players = [p for p in players if (p.get("expert_salute") or 0) >= expert_salute_min]
    if has_ml_data is not None:
        # Raw key is snake_case (merger.py sets `player["has_ml_data"]`) —
        # verified, not "hasMlData" like the scoring.py-computed fields.
        players = [p for p in players if bool(p.get("has_ml_data")) == has_ml_data]
    if has_risk_flag is not None:
        players = [p for p in players if (p.get("rischio") is not None) == has_risk_flag]

    # Sorting — combine up to MAX_SORT_KEYS criteria by applying them from
    # least to most important (see _apply_single_sort docstring).
    sort_keys = _parse_sort_keys(sort_by)
    for field, reverse in reversed(sort_keys):
        _apply_single_sort(players, field, reverse)

    total = len(players)
    start = (page - 1) * size
    items = players[start:start + size]

    camel_items = [{_to_camel(k): v for k, v in p.items()} for p in items]

    return ORJSONResponse({
        "total": total,
        "page": page,
        "size": size,
        "items": camel_items,
        "meta": data.get("meta"),
    })
