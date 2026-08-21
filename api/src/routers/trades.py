"""Trade dashboard + execute — runtime RosterContext only.

No fantasy roster rows are written to PostgreSQL. Transfer results are
audited in process memory only.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from ml.lineup.enrichment import parse_hybrid_rows
from ml.roster_import.matcher import MatchStatus
from ml.roster_import.store import get_default_store
from ml.trades.advisor import TradePlayer, build_trade_dashboard
from ml.trades.credit_penalty import recompute_value_on_transfer
from ml.trades.enrichment import enrich_players, season_notice_if_cold_start
from ml.trades.fairness import evaluate_trade
from ml.trades.signals import MatchdayVote

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


class TradeEvaluateRequest(_CamelModel):
    """Bilateral trade fairness evaluation (read-only simulation)."""

    context_id: str
    sheet_name: str
    team_name: str
    mode: Literal["classic", "mantra"] = "mantra"
    give: list[str] = Field(default_factory=list)  # player_id (fantacalcio_id)
    receive: list[str] = Field(default_factory=list)
    formation_prefs: list[str] = Field(
        default_factory=lambda: ["4-3-3", "3-5-2", "3-4-3"]
    )
    tolerance_percent: float = Field(10.0, ge=0, le=50)


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
    # `db` viene passato a `_load_hybrid_map` per risolvere la stagione
    # corrente da `player_quotations` (vedi commento in `_load_hybrid_map`).
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


# ── Fairness evaluation (read-only) ──────────────────────────────────────────


async def _load_votes_map(
    db: AsyncSession,
    fantacalcio_ids: list[int],
    *,
    season_start: int | None = None,
    window: int = 8,
) -> dict[int, list[MatchdayVote]]:
    """Recent matchday fantavoto rows for the given players."""
    if not fantacalcio_ids:
        return {}

    if season_start is None:
        row = (
            await db.execute(
                sa.text("SELECT MAX(season_start) FROM player_matchday_votes")
            )
        ).scalar_one_or_none()
        season_start = int(row) if row is not None else 0
        if season_start == 0:
            return {}

    placeholders = ", ".join(f":id{i}" for i in range(len(fantacalcio_ids)))
    params: dict[str, Any] = {"ss": season_start}
    for i, fid in enumerate(fantacalcio_ids):
        params[f"id{i}"] = int(fid)

    # Fetch a slightly wider window; EWMA will keep the last N usable games
    sql = f"""
        SELECT fantacalcio_id, giornata, fantavoto
        FROM player_matchday_votes
        WHERE season_start = :ss
          AND fantacalcio_id IN ({placeholders})
          AND fonte = 'fantacalcio'
        ORDER BY giornata DESC
    """
    try:
        result = await db.execute(sa.text(sql), params)
    except Exception as exc:  # noqa: BLE001 — table may not exist yet
        log.warning("player_matchday_votes unavailable: %s", exc)
        return {}

    out: dict[int, list[MatchdayVote]] = {}
    for r in result:
        m = r._mapping
        fid = int(m["fantacalcio_id"])
        fv = m["fantavoto"]
        out.setdefault(fid, []).append(
            MatchdayVote(
                giornata=int(m["giornata"]),
                fantavoto=float(fv) if fv is not None else None,
            )
        )
    # Trim per player
    for fid in out:
        out[fid] = out[fid][:window]
    return out


async def _load_status_map(
    db: AsyncSession,
    fantacalcio_ids: list[int],
) -> dict[int, dict]:
    if not fantacalcio_ids:
        return {}
    placeholders = ", ".join(f":id{i}" for i in range(len(fantacalcio_ids)))
    params: dict[str, Any] = {}
    for i, fid in enumerate(fantacalcio_ids):
        params[f"id{i}"] = int(fid)

    # Latest matchday per player (or global latest)
    sql = f"""
        SELECT DISTINCT ON (fantacalcio_id)
            fantacalcio_id, matchday, status, probability, team
        FROM player_matchday_status
        WHERE fantacalcio_id IN ({placeholders})
        ORDER BY fantacalcio_id, season_start DESC, matchday DESC
    """
    try:
        result = await db.execute(sa.text(sql), params)
    except Exception as exc:  # noqa: BLE001
        log.warning("player_matchday_status unavailable: %s", exc)
        return {}
    return {int(r._mapping["fantacalcio_id"]): dict(r._mapping) for r in result}


async def _load_experts_map(
    db: AsyncSession,
    fantacalcio_ids: list[int],
) -> dict[int, dict]:
    """Latest gruppo_esperti titolarità per player."""
    if not fantacalcio_ids:
        return {}
    # expert_ratings.player_id is stored as "fc-{id}" or plain id
    placeholders = ", ".join(f":id{i}" for i in range(len(fantacalcio_ids)))
    params: dict[str, Any] = {}
    id_list: list[str] = []
    for i, fid in enumerate(fantacalcio_ids):
        key = f"fc-{fid}"
        params[f"id{i}"] = key
        id_list.append(key)
        params[f"raw{i}"] = str(fid)

    # Match both "fc-123" and "123" forms
    raw_ph = ", ".join(f":raw{i}" for i in range(len(fantacalcio_ids)))
    sql = f"""
        SELECT DISTINCT ON (player_id)
            player_id, titolarita, consiglio_esperti_raw, season_start, matchday
        FROM expert_ratings
        WHERE source = 'gruppo_esperti'
          AND (player_id IN ({placeholders}) OR player_id IN ({raw_ph}))
        ORDER BY player_id, season_start DESC, matchday DESC NULLS LAST
    """
    try:
        result = await db.execute(sa.text(sql), params)
    except Exception as exc:  # noqa: BLE001
        log.warning("expert_ratings unavailable: %s", exc)
        return {}

    out: dict[int, dict] = {}
    for r in result:
        m = dict(r._mapping)
        pid = str(m["player_id"])
        if pid.startswith("fc-"):
            fid = int(pid[3:])
        else:
            try:
                fid = int(pid)
            except ValueError:
                continue
        out[fid] = m
    return out


async def _load_classic_roles(
    db: AsyncSession,
    fantacalcio_ids: list[int],
) -> dict[int, str]:
    if not fantacalcio_ids:
        return {}
    placeholders = ", ".join(f":id{i}" for i in range(len(fantacalcio_ids)))
    params: dict[str, Any] = {}
    for i, fid in enumerate(fantacalcio_ids):
        params[f"id{i}"] = int(fid)
    sql = f"""
        SELECT DISTINCT ON (fantacalcio_id)
            fantacalcio_id, role
        FROM player_quotations
        WHERE fantacalcio_id IN ({placeholders})
        ORDER BY fantacalcio_id, season_start DESC
    """
    try:
        result = await db.execute(sa.text(sql), params)
    except Exception as exc:  # noqa: BLE001
        log.warning("player_quotations role lookup failed: %s", exc)
        return {}
    out: dict[int, str] = {}
    for r in result:
        role = (r._mapping["role"] or "").upper()
        if role in ("GK", "DEF", "MID", "FWD"):
            out[int(r._mapping["fantacalcio_id"])] = role
    return out


def _resolve_players_from_context(
    ctx,
    sheet_name: str,
    player_ids: list[str],
    hybrid_map: dict,
    *,
    own_team_name: str | None = None,
) -> tuple[list[TradePlayer], list[str]]:
    """Find TradePlayers for the given ids across all teams in the division."""
    wanted = set(player_ids)
    found: dict[str, TradePlayer] = {}

    teams: list = []
    if own_team_name:
        own = ctx.get_team(sheet_name, own_team_name)
        if own is not None:
            teams.append(own)
    teams.extend(ctx.teams_in_same_division(sheet_name, exclude_team=own_team_name))

    for team in teams:
        for mp in team.players:
            tp = _matched_to_trade_player(mp, hybrid_map)
            if tp and tp.player_id in wanted:
                found[tp.player_id] = tp

    missing = [pid for pid in player_ids if pid not in found]
    ordered = [found[pid] for pid in player_ids if pid in found]
    return ordered, missing


@router.post(
    "/evaluate",
    response_class=ORJSONResponse,
    summary="Evaluate bilateral trade fairness (Classic / Mantra)",
    description=(
        "Read-only simulation. Returns PTV breakdown, validity, coverage impact "
        "and a transparent verdict. Does not modify the roster."
    ),
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def evaluate_trade_endpoint(
    request: Request,
    body: TradeEvaluateRequest,
    db: AsyncSession = Depends(get_db),
) -> ORJSONResponse:
    if not body.give and not body.receive:
        raise HTTPException(
            status_code=400,
            detail="Specifica almeno un giocatore in give o receive",
        )

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

    hybrid_map = await _load_hybrid_map(request, db)

    # Build current roster (for Mantra coverage impact)
    roster: list[TradePlayer] = []
    for mp in team.players:
        tp = _matched_to_trade_player(mp, hybrid_map)
        if tp:
            roster.append(tp)

    # Resolve give / receive players (may live on other teams in the context)
    all_ids = list(dict.fromkeys(body.give + body.receive))
    resolved, missing = _resolve_players_from_context(
        ctx,
        body.sheet_name,
        all_ids,
        hybrid_map,
        own_team_name=body.team_name,
    )
    by_id = {p.player_id: p for p in resolved}

    # Also accept receive players that are free agents: try hybrid-only stub
    for pid in missing:
        try:
            fid = int(pid)
        except ValueError:
            continue
        fp = _fp_from_hybrid(hybrid_map, fid)
        by_id[pid] = TradePlayer(
            player_id=pid,
            name=f"Player {pid}",
            eligible_roles=frozenset({"C"}),
            fp_corr=fp,
        )
        missing = [m for m in missing if m != pid]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Giocatori non risolti nel context: {', '.join(missing)}",
        )

    give_players = [by_id[pid] for pid in body.give if pid in by_id]
    recv_players = [by_id[pid] for pid in body.receive if pid in by_id]

    # Load live signals
    fids = []
    for p in give_players + recv_players + roster:
        try:
            fids.append(int(p.player_id))
        except ValueError:
            pass
    fids = list(dict.fromkeys(fids))

    votes_map = await _load_votes_map(db, fids)
    status_map = await _load_status_map(db, fids)
    experts_map = await _load_experts_map(db, fids)
    classic_roles = await _load_classic_roles(db, fids)

    give_enriched = enrich_players(
        give_players,
        votes_by_fid=votes_map,
        status_by_fid=status_map,
        experts_by_fid=experts_map,
        classic_role_by_fid=classic_roles,
    )
    recv_enriched = enrich_players(
        recv_players,
        votes_by_fid=votes_map,
        status_by_fid=status_map,
        experts_by_fid=experts_map,
        classic_role_by_fid=classic_roles,
    )

    notice = season_notice_if_cold_start(
        votes_map, [p.player_id for p in give_players + recv_players]
    )

    evaluation = evaluate_trade(
        mode=body.mode,
        give=give_enriched,
        receive=recv_enriched,
        current_roster=roster if body.mode == "mantra" else None,
        formation_prefs=body.formation_prefs,
        tolerance_percent=body.tolerance_percent,
        season_notice=notice,
    )

    def _player_view(v) -> dict:
        return {
            "playerId": v.player_id,
            "name": v.name,
            "ptv": v.ptv,
            "confidence": v.confidence,
            "flags": list(v.flags),
            "classicRole": v.classic_role,
            "breakdown": v.breakdown,
        }

    payload: dict[str, Any] = {
        "mode": evaluation.mode,
        "valid": evaluation.valid,
        "validationErrors": list(evaluation.validation_errors),
        "verdict": evaluation.verdict,
        "valueDeltaPercent": evaluation.value_delta_percent,
        "toleranceBandPercent": evaluation.tolerance_band_percent,
        "give": [_player_view(v) for v in evaluation.give],
        "receive": [_player_view(v) for v in evaluation.receive],
        "rationale": list(evaluation.rationale),
        "seasonNotice": evaluation.season_notice,
    }
    if evaluation.squad_impact is not None:
        payload["squadImpact"] = {
            "coverageBefore": evaluation.squad_impact.coverage_before,
            "coverageAfter": evaluation.squad_impact.coverage_after,
            "warning": evaluation.squad_impact.warning,
        }
    return ORJSONResponse(payload)
