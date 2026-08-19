"""Lineup optimize endpoint — consumes runtime RosterContext.

Enriches candidates with hybrid predictions + matchday starter probability
when available; falls back to neutral baseline otherwise.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from ml.lineup.enrichment import (
    EnrichmentStats,
    enrich_matched_players,
    parse_hybrid_rows,
    parse_matchday_rows,
)
from ml.lineup.optimizer import OptimizeResult, optimize_lineup
from ml.roster_import.matcher import MatchStatus
from ml.roster_import.store import get_default_store

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, require_role
from ..schemas import _CamelModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/lineup", tags=["lineup"])


# ── Request / response schemas ───────────────────────────────────────────────


class LineupOptimizeRequest(_CamelModel):
    context_id: str
    sheet_name: str
    team_name: str
    matchday: Optional[int] = Field(None, ge=1, le=40)
    opponent_sheet_name: Optional[str] = None
    opponent_team_name: Optional[str] = None
    ruleset: str = "MANTRA"
    candidate_formations: Optional[list[str]] = None
    min_starter_prob: float = Field(0.15, ge=0.0, le=1.0)


class SlotAssignmentSchema(_CamelModel):
    slot_label: str
    slot_roles: list[str]
    player_id: str
    player_name: str
    expected_score: float
    starter_probability: float
    breakdown_note: str = ""


class FormationAlternativeSchema(_CamelModel):
    formation: str
    feasible: bool
    score_totale: float = 0.0
    reason: str = ""


class LineupOptimizeResponse(_CamelModel):
    context_id: str
    team_name: str
    sheet_name: str
    matchday: Optional[int] = None
    chosen_formation: Optional[str] = None
    score_totale: Optional[float] = None
    starting_xi: list[SlotAssignmentSchema] = Field(default_factory=list)
    bench: list[SlotAssignmentSchema] = Field(default_factory=list)
    alternatives_considered: list[FormationAlternativeSchema] = Field(
        default_factory=list
    )
    opponent_head_to_head: Optional[dict] = None
    enrichment: Optional[dict] = None
    notes: list[str] = Field(default_factory=list)


def _get_store(request: Request):
    store = getattr(request.app.state, "roster_store", None)
    if store is None:
        store = get_default_store()
        request.app.state.roster_store = store
    return store


def _get_repo(request: Request) -> DataRepository:
    repo: DataRepository | None = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Data repository not initialised")
    return repo


async def _load_enrichment_maps(
    request: Request,
    db: AsyncSession,
    *,
    matchday: int | None,
    fantacalcio_ids: list[int],
) -> tuple[dict, dict, int | None, list[str]]:
    """Load hybrid + matchday maps; never raises — degrades gracefully."""
    notes: list[str] = []
    hybrid_map: dict = {}
    matchday_map: dict = {}
    resolved_md: int | None = matchday
    repo = _get_repo(request)

    # Hybrid predictions (artifact — may be missing)
    try:
        # Pass `db` so get_hybrid_predictions() resolves the current season
        # from player_quotations instead of falling back to the hardcoded
        # 2025 default — the 2025 artefact would silently load stale data
        # in 2026-27.
        hybrid_data = await repo.get_hybrid_predictions(db=db)
        rows: list = []
        if isinstance(hybrid_data, dict):
            rows = (
                hybrid_data.get("players")
                or hybrid_data.get("results")
                or hybrid_data.get("items")
                or []
            )
            if not rows and any(
                isinstance(v, dict) and "fantacalcio_id" in v
                for v in hybrid_data.values()
            ):
                rows = list(hybrid_data.values())
        elif isinstance(hybrid_data, list):
            rows = hybrid_data
        hybrid_map = parse_hybrid_rows(rows)
        if not hybrid_map:
            notes.append("Hybrid predictions vuote — FP baseline")
    except Exception as exc:  # noqa: BLE001
        log.warning("Hybrid predictions unavailable: %s", exc)
        notes.append("Hybrid predictions non disponibili — FP baseline")

    # Matchday status (DB)
    try:
        resolved_md, md_rows = await repo.load_matchday_status_bulk(
            db, matchday=matchday, fantacalcio_ids=fantacalcio_ids or None
        )
        matchday_map = parse_matchday_rows(md_rows)
        if not matchday_map:
            notes.append(
                f"Nessuno status matchday per giornata {resolved_md} — SP di default"
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Matchday status unavailable: %s", exc)
        notes.append("Matchday status non disponibile — SP di default")
        resolved_md = matchday

    return hybrid_map, matchday_map, resolved_md, notes


def _result_to_response(
    *,
    context_id: str,
    sheet_name: str,
    team_name: str,
    matchday: int | None,
    result: OptimizeResult,
    notes: list[str],
    opponent_h2h: dict | None,
    enrichment_stats: EnrichmentStats | None,
) -> LineupOptimizeResponse:
    starting: list[SlotAssignmentSchema] = []
    if result.chosen and result.chosen.gk:
        g = result.chosen.gk
        starting.append(
            SlotAssignmentSchema(
                slot_label=g.slot_label,
                slot_roles=sorted(g.slot_roles),
                player_id=g.player_id,
                player_name=g.player_name,
                expected_score=g.expected_value,
                starter_probability=g.starter_probability,
                breakdown_note=g.breakdown_note,
            )
        )
    if result.chosen:
        for a in result.chosen.assignments:
            starting.append(
                SlotAssignmentSchema(
                    slot_label=a.slot_label,
                    slot_roles=sorted(a.slot_roles),
                    player_id=a.player_id,
                    player_name=a.player_name,
                    expected_score=a.expected_value,
                    starter_probability=a.starter_probability,
                    breakdown_note=a.breakdown_note,
                )
            )

    bench = [
        SlotAssignmentSchema(
            slot_label="bench",
            slot_roles=sorted(c.eligible_roles),
            player_id=c.player_id,
            player_name=c.name,
            expected_score=round(c.expected_value, 4),
            starter_probability=c.starter_probability,
            breakdown_note=c.breakdown_note,
        )
        for c in result.bench
    ]

    alts = [
        FormationAlternativeSchema(
            formation=a.formation,
            feasible=a.feasible,
            score_totale=a.score_totale,
            reason=a.reason,
        )
        for a in result.alternatives
    ]

    enrichment = None
    if enrichment_stats is not None:
        enrichment = {
            "candidates": enrichment_stats.total,
            "withHybrid": enrichment_stats.with_hybrid,
            "withMatchday": enrichment_stats.with_matchday,
            "excludedOut": enrichment_stats.excluded_out,
            "baselineFallback": enrichment_stats.baseline_fallback,
        }

    return LineupOptimizeResponse(
        context_id=context_id,
        team_name=team_name,
        sheet_name=sheet_name,
        matchday=matchday,
        chosen_formation=result.chosen.formation if result.chosen else None,
        score_totale=result.chosen.score_totale if result.chosen else None,
        starting_xi=starting,
        bench=bench,
        alternatives_considered=alts,
        opponent_head_to_head=opponent_h2h,
        enrichment=enrichment,
        notes=notes,
    )


@router.post(
    "/optimize",
    response_model=LineupOptimizeResponse,
    response_class=ORJSONResponse,
    summary="Optimize starting XI for a matchday",
    description=(
        "Exact assignment over official Mantra modules (Hungarian algorithm). "
        "EV = FP_Ibrido × StarterProbability × OpponentAdjustment when hybrid "
        "and matchday data are available; otherwise neutral baseline. "
        "Requires a live RosterContext from POST /roster/import. "
        "Opponent (same division only) is optional and informational."
    ),
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def optimize(
    request: Request,
    body: LineupOptimizeRequest,
    db: AsyncSession = Depends(get_db),
) -> LineupOptimizeResponse:
    if body.ruleset.upper() != "MANTRA":
        raise HTTPException(
            status_code=400,
            detail="Solo ruleset MANTRA è supportato in questa versione",
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
        raise HTTPException(
            status_code=404,
            detail=f"Squadra '{body.team_name}' non trovata in '{body.sheet_name}'",
        )
    if team.is_empty:
        raise HTTPException(status_code=400, detail="La squadra selezionata è vuota")

    # Collect fantacalcio ids for bulk enrichment
    fids: list[int] = []
    for mp in team.players:
        if mp.catalog is not None:
            fids.append(int(mp.catalog.fantacalcio_id))

    hybrid_map, matchday_map, resolved_md, enrich_notes = await _load_enrichment_maps(
        request, db, matchday=body.matchday, fantacalcio_ids=fids
    )

    candidates, stats = enrich_matched_players(
        team.players,
        hybrid_by_fid=hybrid_map,
        matchday_by_fid=matchday_map,
    )

    notes = list(enrich_notes)
    if stats.excluded_out:
        notes.append(
            f"{stats.excluded_out} giocatori esclusi (infortunati/squalificati)"
        )
    if stats.baseline_fallback and stats.with_hybrid == 0:
        notes.append("Tutti gli EV usano FP baseline (hybrid assente)")

    # Opponent head-to-head (same division only)
    opponent_h2h: dict | None = None
    if body.opponent_team_name:
        opp_sheet = body.opponent_sheet_name or body.sheet_name
        if opp_sheet != body.sheet_name:
            raise HTTPException(
                status_code=422,
                detail="L'avversario deve appartenere alla stessa divisione",
            )
        opp = ctx.get_team(opp_sheet, body.opponent_team_name)
        if opp is None:
            raise HTTPException(
                status_code=404,
                detail=f"Avversario '{body.opponent_team_name}' non trovato",
            )
        if opp.is_empty:
            opponent_h2h = {
                "opponentTeamName": body.opponent_team_name,
                "opponentDataAvailable": False,
            }
            notes.append(
                "Rosa avversaria vuota — confronto testa a testa non disponibile"
            )
        else:
            opp_fids = [
                int(mp.catalog.fantacalcio_id)
                for mp in opp.players
                if mp.catalog is not None
            ]
            # reuse maps; load extra matchday if needed
            opp_candidates, _ = enrich_matched_players(
                opp.players,
                hybrid_by_fid=hybrid_map,
                matchday_by_fid=matchday_map,
            )
            opp_candidates.sort(key=lambda c: c.expected_value, reverse=True)
            top11 = opp_candidates[:11]
            opponent_h2h = {
                "opponentTeamName": body.opponent_team_name,
                "opponentDataAvailable": True,
                "expectedTotalScore": round(
                    sum(c.expected_value for c in top11), 2
                ),
                "probableXiCount": len(top11),
                "unmatchedPlayersExcluded": len(opp.players)
                - sum(
                    1
                    for mp in opp.players
                    if mp.status
                    in (MatchStatus.AUTO, MatchStatus.PROVISIONAL, MatchStatus.MANUAL)
                ),
            }

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="Nessun giocatore eleggibile (matchati e non out) per l'ottimizzazione",
        )

    result = optimize_lineup(
        candidates,
        formations=body.candidate_formations,
        min_starter_prob=body.min_starter_prob,
    )

    if result.chosen is None:
        notes.append("Nessun modulo fattibile con la rosa e le soglie attuali")

    return _result_to_response(
        context_id=body.context_id,
        sheet_name=body.sheet_name,
        team_name=body.team_name,
        matchday=resolved_md,
        result=result,
        notes=notes,
        opponent_h2h=opponent_h2h,
        enrichment_stats=stats,
    )
