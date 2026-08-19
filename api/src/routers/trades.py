"""Trade dashboard + execute — runtime RosterContext only.

No fantasy roster rows are written to PostgreSQL. Transfer results are
audited in process memory only.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from ml.lineup.enrichment import parse_hybrid_rows
from ml.roster_import.matcher import MatchStatus
from ml.roster_import.store import get_default_store
from ml.trades.advisor import TradePlayer, build_trade_dashboard
from ml.trades.credit_penalty import recompute_value_on_transfer

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, require_role
from ..schemas import _CamelModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/trades", tags=["trades"])


class TradesDashboardRequest(_CamelModel):
    context_id: str
    sheet_name: str
    team_name: str
    formation_prefs: list[str] = Field(
        default_factory=lambda: ["4-3-3", "3-5-2", "3-4-3"]
    )
    hard_exclusion_threshold: float = Field(75.0, ge=0, le=200)


class CreditPenaltyPreviewRequest(_CamelModel):
    original_purchase_price: int = Field(..., ge=0)
    current_value: int = Field(..., ge=0)
    decay_step_percent: float = Field(25.0, ge=0, le=100)
    floor_percent: float = Field(25.0, ge=0, le=100)


class TradeLegSchema(_CamelModel):
    player_id: str
    original_purchase_price: int = Field(..., ge=0)
    current_value: int = Field(..., ge=0)


class TradeExecuteRequest(_CamelModel):
    context_id: str
    sheet_name: str
    from_team_name: str
    to_team_name: str
    give: list[TradeLegSchema] = Field(default_factory=list)
    receive: list[TradeLegSchema] = Field(default_factory=list)
    credit_penalty_enabled: bool = False
    decay_step_percent: float = Field(25.0, ge=0, le=100)
    floor_percent: float = Field(25.0, ge=0, le=100)


def _get_store(request: Request):
    store = getattr(request.app.state, "roster_store", None)
    if store is None:
        store = get_default_store()
        request.app.state.roster_store = store
    return store


def _get_repo(request: Request) -> DataRepository | None:
    return getattr(request.app.state, "repo", None)


async def _load_hybrid_map(request: Request, db: AsyncSession) -> dict[int, Any]:
    repo = _get_repo(request)
    if repo is None:
        return {}
    try:
        # Pass `db` so get_hybrid_predictions() resolves the current season
        # from player_quotations instead of falling back to the hardcoded
        # 2025 default — see lineup._load_enrichment_maps for context.
        hybrid_data = await repo.get_hybrid_predictions(db=db)
        rows: list = []
        if isinstance(hybrid_data, dict):
            rows = (
                hybrid_data.get("players")
                or hybrid_data.get("results")
                or hybrid_data.get("items")
                or []
            )
        elif isinstance(hybrid_data, list):
            rows = hybrid_data
        return parse_hybrid_rows(rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("Hybrid unavailable for trades: %s", exc)
        return {}


def _fp_from_hybrid(hybrid_map: dict, fid: int) -> float:
    info = hybrid_map.get(fid)
    if info is None:
        return 50.0
    fp = float(info.fp_ibrido_voto)
    if fp > 12:
        return max(0.0, min(100.0, fp))
    return max(0.0, min(100.0, (fp - 4.0) / 6.0 * 100.0))


def _matched_to_trade_player(mp, hybrid_map: dict | None = None) -> TradePlayer | None:
    if mp.status == MatchStatus.UNMATCHED or mp.catalog is None:
        return None
    roles = mp.catalog.roles_mantra
    if not roles:
        classic = (mp.catalog.role_classic or "").upper()
        mapping = {"P": ("Por",), "D": ("Dc",), "C": ("C",), "A": ("A",)}
        roles = mapping.get(classic, ())
    if not roles:
        return None
    fid = int(mp.catalog.fantacalcio_id)
    fp = _fp_from_hybrid(hybrid_map or {}, fid)
    return TradePlayer(
        player_id=str(fid),
        name=mp.catalog.name or mp.parsed.name_clean,
        eligible_roles=frozenset(roles),
        cost=mp.parsed.cost,
        current_value=mp.parsed.cost,
        fp_corr=fp,
        team_serie_a=mp.catalog.team or "",
    )


def _team_player_ids(team) -> set[str]:
    ids: set[str] = set()
    for mp in team.players:
        if mp.catalog is not None:
            ids.add(str(mp.catalog.fantacalcio_id))
    return ids


def _player_json(p: TradePlayer) -> dict:
    return {
        "playerId": p.player_id,
        "name": p.name,
        "roles": sorted(p.eligible_roles),
        "cost": p.cost,
        "currentValue": p.current_value,
        "fpCorr": round(p.fp_corr, 2),
        "teamSerieA": p.team_serie_a,
    }


_transfer_log: list[dict] = []


def _log_transfer(record: dict) -> None:
    _transfer_log.append(record)
    if len(_transfer_log) > 500:
        del _transfer_log[:-500]


@router.post(
    "/dashboard",
    response_class=ORJSONResponse,
    summary="Trade advisor dashboard",
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def trades_dashboard(
    request: Request,
    body: TradesDashboardRequest,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    del db
    store = _get_store(request)
    ctx = store.get(body.context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )
    team = ctx.get_team(body.sheet_name, body.team_name)
    if team is None:
        raise HTTPException(status_code=404, detail="Squadra non trovata")
    if team.is_empty:
        raise HTTPException(status_code=400, detail="Squadra vuota")

    hybrid_map = await _load_hybrid_map(request, db)
    notes: list[str] = []
    if not hybrid_map:
        notes.append("Hybrid non disponibile — FP_Corr baseline 50")

    squad: list[TradePlayer] = []
    for mp in team.players:
        tp = _matched_to_trade_player(mp, hybrid_map)
        if tp:
            squad.append(tp)

    if not squad:
        raise HTTPException(
            status_code=400,
            detail="Nessun giocatore matchato disponibile per l'analisi scambi",
        )

    market: list[TradePlayer] = []
    for other in ctx.teams_in_same_division(
        body.sheet_name, exclude_team=body.team_name
    ):
        for mp in other.players:
            tp = _matched_to_trade_player(mp, hybrid_map)
            if tp:
                market.append(tp)

    dash = build_trade_dashboard(
        squad,
        body.formation_prefs,
        market_pool=market,
        hard_exclusion_threshold=body.hard_exclusion_threshold,
    )

    return ORJSONResponse(
        {
            "contextId": body.context_id,
            "teamName": body.team_name,
            "sheetName": body.sheet_name,
            "formationPrefs": list(dash.formation_prefs),
            "coverageByFormation": dash.coverage_by_formation,
            "coverageMatrix": [
                {
                    "formation": c.formation,
                    "slotLabel": c.slot_label,
                    "status": c.status,
                    "missing": c.missing,
                }
                for c in dash.coverage_cells
            ],
            "tradeOutCandidates": [
                {
                    "player": _player_json(c.player),
                    "retentionScore": round(c.retention, 2),
                    "surplusRoles": list(c.surplus_roles),
                    "rationale": c.rationale,
                }
                for c in dash.trade_out
            ],
            "tradeInTargets": [
                {
                    "playerId": t.player_id,
                    "name": t.name,
                    "coversSlots": list(t.covers_slots),
                    "fpCorr": t.fp_corr,
                    "estimatedCost": t.estimated_cost,
                    "roles": list(t.roles),
                }
                for t in dash.trade_in
            ],
            "excludedTopPerformers": [
                {
                    "player": _player_json(e.player),
                    "retentionScore": round(e.retention, 2),
                    "reason": e.reason,
                }
                for e in dash.excluded_top_performers
            ],
            "notes": notes
            + ["Market pool = altre rose della stessa divisione nel context"],
        }
    )


@router.post(
    "/credit-penalty/preview",
    response_class=ORJSONResponse,
    summary="Preview credit penalty on a transfer",
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def credit_penalty_preview(body: CreditPenaltyPreviewRequest) -> ORJSONResponse:
    new_value = recompute_value_on_transfer(
        body.original_purchase_price,
        body.current_value,
        body.decay_step_percent,
        body.floor_percent,
    )
    return ORJSONResponse(
        {
            "originalPurchasePrice": body.original_purchase_price,
            "currentValue": body.current_value,
            "newValue": new_value,
            "decayStepPercent": body.decay_step_percent,
            "floorPercent": body.floor_percent,
        }
    )


@router.post(
    "/execute",
    response_class=ORJSONResponse,
    summary="Register a trade (runtime-only audit)",
    description=(
        "Validates both teams share the same division context, optionally "
        "applies credit penalty, writes an in-memory audit record. "
        "Does not persist roster changes to PostgreSQL."
    ),
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def execute_trade(
    request: Request,
    body: TradeExecuteRequest,
) -> ORJSONResponse:
    if not body.give and not body.receive:
        raise HTTPException(
            status_code=400,
            detail="Specifica almeno un giocatore in give o receive",
        )
    if body.from_team_name == body.to_team_name:
        raise HTTPException(status_code=400, detail="Le due squadre devono essere diverse")

    store = _get_store(request)
    ctx = store.get(body.context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )

    from_team = ctx.get_team(body.sheet_name, body.from_team_name)
    to_team = ctx.get_team(body.sheet_name, body.to_team_name)
    if from_team is None or to_team is None:
        raise HTTPException(
            status_code=404,
            detail="Una o entrambe le squadre non trovate nella stessa divisione",
        )

    from_ids = _team_player_ids(from_team)
    to_ids = _team_player_ids(to_team)
    legs_out: list[dict] = []

    def _process_leg(leg: TradeLegSchema, owner_ids: set[str], direction: str) -> dict:
        if leg.player_id not in owner_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Giocatore {leg.player_id} non appartiene alla rosa "
                    f"di partenza ({direction})"
                ),
            )
        value_after = leg.current_value
        if body.credit_penalty_enabled:
            value_after = recompute_value_on_transfer(
                leg.original_purchase_price,
                leg.current_value,
                body.decay_step_percent,
                body.floor_percent,
            )
        return {
            "playerId": leg.player_id,
            "direction": direction,
            "valueBefore": leg.current_value,
            "valueAfter": value_after,
            "originalPurchasePrice": leg.original_purchase_price,
            "penaltyApplied": body.credit_penalty_enabled,
        }

    for leg in body.give:
        legs_out.append(_process_leg(leg, from_ids, "give"))
    for leg in body.receive:
        legs_out.append(_process_leg(leg, to_ids, "receive"))

    transfer_id = str(uuid.uuid4())
    record = {
        "transferId": transfer_id,
        "contextId": body.context_id,
        "sheetName": body.sheet_name,
        "fromTeamName": body.from_team_name,
        "toTeamName": body.to_team_name,
        "legs": legs_out,
        "creditPenaltyEnabled": body.credit_penalty_enabled,
        "recordedAt": time.time(),
        "notes": [
            "Audit in-memory only — nessuna modifica rosa su PostgreSQL",
            "Il client deve aggiornare lo stato locale della rosa dopo lo scambio",
        ],
    }
    _log_transfer(record)
    return ORJSONResponse(record)
