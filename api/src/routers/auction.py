"""HTTP API for the live auction tracker.

Exposes the :mod:`ml.auction` stateful orchestrator as a REST interface,
preserving its single-operator, single-process semantics:

* Sessions live in ``app.state.auction_sessions`` (an in-process dict);
  each session is identified by a server-generated ``session_id``.
* The client owns the lifecycle: ``init`` → ``record`` / ``undo`` /
  ``projection`` / ``alternatives`` / ``summary`` / ``serialize`` →
  ``discard`` (or rely on server restart to clear all sessions).
* All endpoints require a valid JWT with at least the ``member`` role.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..data_repository import DataRepository
from ..deps import get_db, require_role
from ..schemas import (
    AlternativesConfigSchema,
    AlternativesResponse,
    AssignmentRecordSchema,
    AuctionConfigSchema,
    AuctionParticipantSetupSchema,
    AuctionPlayerSchema,
    AuctionPlayerSummarySchema,
    AuctionParticipantStateSchema,
    AuctionSimulationResponse,
    AuctionSummarySchema,
    BidderPolicySchema,
    BidderProfileSchema,
    FormationCoverageSchema,
    InitializeAuctionRequest,
    InitializeAuctionResponse,
    MarketDriftConfigSchema,
    ParticipantSimStatsSchema,
    PlayerAcquisitionStatsSchema,
    ProjectionResponse,
    RecordAssignmentRequest,
    RecordAssignmentResponse,
    SerializedAuctionStateResponse,
    SimulateAuctionRequest,
    VarRankingItemSchema,
    VarRankingResponse,
)
from ml.auction.models import (
    AlternativesConfig,
    AuctionConfig,
    MarketDriftConfig,
    ParticipantSetup,
    Role,
    Tier,
)
from ml.auction.orchestrator import (
    AuctionSession,
    deserialize_state,
)
from ml.auction.price_drift import classify_tier, get_current_projection
from ml.auction.models import ValuationMode
from ml.auction.simulation import (
    AuctionSimulationConfig,
    BidderPolicy,
    BidderProfile,
    simulate_auction,
)
from ml.auction.var import VarEngine
from ml.optimizer.inflation import InflationConfig
from ml.optimizer.models import Player, RulesetType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auction", tags=["auction"])


# ---------------------------------------------------------------------------
# Type-safe mappers: Pydantic ↔ internal dataclasses
# ---------------------------------------------------------------------------


def _player_from_schema(p: AuctionPlayerSchema) -> Player:
    eligible = frozenset(p.eligible_roles) if p.eligible_roles else frozenset()
    return Player(
        player_id=p.player_id,
        name=p.name,
        role=cast(Role, p.role),
        real_team=p.real_team,
        cost=p.cost,
        projected_score=p.projected_score,
        season_value=p.season_value,
        start_probability=p.start_probability,
        eligible_roles=eligible,
        reliability_weight=p.reliability_weight,
        sample_cohort=p.sample_cohort,
    )


def _player_to_summary(p: Player) -> AuctionPlayerSummarySchema:
    return AuctionPlayerSummarySchema(
        player_id=p.player_id,
        name=p.name,
        real_team=p.real_team,
        role=p.role,
        cost=p.cost,
        projected_score=p.projected_score,
        season_value=p.season_value,
        start_probability=p.start_probability,
        eligible_roles=sorted(p.eligible_roles) if p.eligible_roles else None,
        sample_cohort=p.sample_cohort,
        reliability_weight=p.reliability_weight,
    )


def _participant_from_schema(p: AuctionParticipantSetupSchema) -> ParticipantSetup:
    return ParticipantSetup(
        participant_id=p.participant_id,
        display_name=p.display_name,
        budget_initial=p.budget_initial,
    )


def _market_drift_from_schema(cfg: MarketDriftConfigSchema) -> MarketDriftConfig:
    return MarketDriftConfig(
        alpha=cfg.alpha,
        spillover_adjacent_tier=cfg.spillover_adjacent_tier,
        spillover_cross_role=cfg.spillover_cross_role,
        min_index=cfg.min_index,
        max_index=cfg.max_index,
        tier_thresholds=(float(cfg.tier_thresholds[0]), float(cfg.tier_thresholds[1])),
    )


def _alternatives_config_from_schema(
    cfg: AlternativesConfigSchema | None,
) -> AlternativesConfig:
    if cfg is None:
        return AlternativesConfig()
    return AlternativesConfig(low_cost_percentile=cfg.low_cost_percentile)


def _auction_config_from_schema(
    cfg: AuctionConfigSchema,
    inflation: InflationConfig | None,
) -> AuctionConfig:
    return AuctionConfig(
        num_participants=cfg.num_participants,
        role_quotas=dict(cfg.role_quotas),
        ruleset=cast(RulesetType, cfg.ruleset),
        market_drift_config=_market_drift_from_schema(cfg.market_drift_config),
        alternatives_config=_alternatives_config_from_schema(cfg.alternatives_config),
        use_inflation_baseline=cfg.use_inflation_baseline,
        inflation_config=inflation,
        valuation_mode=cfg.valuation_mode,
        hybrid_blend=getattr(cfg, "hybrid_blend", 0.0) or 0.0,
        reference_budget=cfg.reference_budget,
        budget_initial=cfg.budget_initial,
    )


def _participant_to_schema(p: object) -> AuctionParticipantStateSchema:
    """Cast un ParticipantState generico a schema Pydantic (typing-safe)."""
    from ml.auction.models import ParticipantState

    state = cast(ParticipantState, p)
    return AuctionParticipantStateSchema(
        participant_id=state.participant_id,
        display_name=state.display_name,
        budget_residual=state.budget_residual,
        squad=[_player_to_summary(pl) for pl in state.squad],
        role_breakdown={role: int(n) for role, n in state.role_breakdown.items()},
    )


def _assignment_to_schema(a: object) -> AssignmentRecordSchema:
    """Cast un AssignmentRecord generico a schema Pydantic (typing-safe)."""
    from ml.auction.models import AssignmentRecord

    rec = cast(AssignmentRecord, a)
    return AssignmentRecordSchema(
        sequence_number=rec.sequence_number,
        player=_player_to_summary(rec.player),
        winner_participant_id=rec.winner_participant_id,
        final_price=rec.final_price,
        role=rec.role,
        tier=rec.tier,
        price_index_before=rec.price_index_before,
        price_index_after=rec.price_index_after,
        assigned_slot=rec.assigned_slot if rec.assigned_slot is not None else rec.role,
    )


def _price_index_to_dict(
    price_index: dict[Role, dict[Tier, float]],
) -> dict[str, dict[str, float]]:
    return {role: {tier: float(v) for tier, v in tiers.items()} for role, tiers in price_index.items()}


def _mantra_coverage_to_schema(
    coverage: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, FormationCoverageSchema]] | None:
    """Translate domain FormationCoverage map to API schemas."""
    if coverage is None:
        return None
    out: dict[str, dict[str, FormationCoverageSchema]] = {}
    for pid, modules in coverage.items():
        out[pid] = {}
        for label, cov in modules.items():
            out[pid][label] = FormationCoverageSchema(
                label=getattr(cov, "label", label),
                feasible=bool(getattr(cov, "feasible", False)),
                deficits=dict(getattr(cov, "deficits", {}) or {}),
                assigned=getattr(cov, "assigned", None),
            )
    return out


def _summary_to_schema(summary: object) -> AuctionSummarySchema:
    from ml.auction.models import AuctionSummary

    s = cast(AuctionSummary, summary)
    return AuctionSummarySchema(
        participants=[_participant_to_schema(p) for p in s.participants],
        assignments=[_assignment_to_schema(a) for a in s.assignments],
        price_index=_price_index_to_dict(s.price_index),
        completion_probability=s.completion_probability,
        mantra_module_coverage=_mantra_coverage_to_schema(s.mantra_module_coverage),
    )



def _bidder_policy_from_schema(p: BidderPolicySchema) -> BidderPolicy:
    return BidderPolicy(
        aggressiveness=p.aggressiveness,
        inflation_tolerance=p.inflation_tolerance,
        max_overpay_ratio=p.max_overpay_ratio,
        min_residual_credits_per_slot=p.min_residual_credits_per_slot,
        all_in_probability=p.all_in_probability,
        budget_elasticity=p.budget_elasticity,
        var_weight=p.var_weight,
        team_strength_weight=p.team_strength_weight,
        prefer_alternatives=p.prefer_alternatives,
        prefer_low_cost_alternative=p.prefer_low_cost_alternative,
        rebid_trigger_pct_above_expected=p.rebid_trigger_pct_above_expected,
        budget_share_by_role=p.budget_share_by_role,
        phase_bias=p.phase_bias,
        prefer_young_players=p.prefer_young_players,
        max_age_preference=p.max_age_preference,
        prefer_high_start_probability=p.prefer_high_start_probability,
        min_start_probability=p.min_start_probability,
        prefer_high_variance=p.prefer_high_variance,
        prefer_multi_role=p.prefer_multi_role,
        min_num_roles=p.min_num_roles,
        budget_share_by_block=p.budget_share_by_block,
        max_top_tier_count=p.max_top_tier_count,
        target_top_tier_count=p.target_top_tier_count,
        avoid_top_tier_early=p.avoid_top_tier_early,
        adaptive=p.adaptive,
        adapt_on=tuple(p.adapt_on or ()),
    )


def _bidder_profile_from_schema(p: BidderProfileSchema) -> BidderProfile:
    return BidderProfile(participant_id=p.participant_id, policy=_bidder_policy_from_schema(p.policy))


def _sim_config_from_schema(cfg: object) -> AuctionSimulationConfig:
    from ..schemas import AuctionSimulationConfigSchema
    c = cast(AuctionSimulationConfigSchema, cfg)
    return AuctionSimulationConfig(
        n_simulations=c.n_simulations, random_seed=c.random_seed,
        price_noise_std_ratio=c.price_noise_std_ratio, timeout_seconds=c.timeout_seconds,
        min_bid_step=c.min_bid_step,
    )


def _resolve_inflation(cfg: AuctionConfigSchema) -> InflationConfig | None:
    if not cfg.use_inflation_baseline:
        return None
    if cfg.inflation_config is not None:
        ic = cfg.inflation_config
        return InflationConfig(
            inflation_percentile_threshold=ic.inflation_percentile_threshold,
            max_inflation_multiplier=ic.max_inflation_multiplier,
            base_inflation_rate=ic.base_inflation_rate,
            baseline_participants=ic.baseline_participants,
            team_strength_multiplier=ic.team_strength_multiplier,
        )
    return InflationConfig()


# ---------------------------------------------------------------------------
# Session store helpers
# ---------------------------------------------------------------------------


def _get_session_store(request: Request) -> dict[str, AuctionSession]:
    store = getattr(request.app.state, "auction_sessions", None)
    if store is None:
        store = {}
        request.app.state.auction_sessions = store
    return store


def _get_session(request: Request, session_id: str) -> AuctionSession:
    store = _get_session_store(request)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session_id {session_id!r} not found",
        )
    return session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/init",
    response_model=InitializeAuctionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("member"))],
)
async def init_auction(
    payload: InitializeAuctionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> InitializeAuctionResponse:
    """Inizializza una nuova sessione d'asta, ritorna il ``session_id``.

    Il pool di giocatori può essere fornito dal client tramite
    ``player_pool`` oppure costruito lato server dal DB + predizioni
    ML, replicando il pattern di ``POST /optimize/multi``: in
    particolare viene chiamato
    :meth:`DataRepository.get_player_pool` con ``min_qt_a=1`` (tutti
    i giocatori con quotazione disponibile per la stagione).  Se
    anche il pool dal DB è vuoto, l'endpoint risponde 400 con
    dettaglio esplicito per il client.
    """
    inflation: InflationConfig | None = None
    if payload.config.use_inflation_baseline:
        if payload.config.inflation_config is not None:
            ic = payload.config.inflation_config
            inflation = InflationConfig(
                inflation_percentile_threshold=ic.inflation_percentile_threshold,
                max_inflation_multiplier=ic.max_inflation_multiplier,
                base_inflation_rate=ic.base_inflation_rate,
                baseline_participants=ic.baseline_participants,
                team_strength_multiplier=ic.team_strength_multiplier,
            )
        else:
            inflation = InflationConfig()

    auction_cfg = _auction_config_from_schema(payload.config, inflation)
    participants = [_participant_from_schema(p) for p in payload.participants]

    n_excluded_no_projection = 0
    if payload.player_pool is not None:
        pool: list[Player] = [_player_from_schema(p) for p in payload.player_pool]
    else:
        repo = DataRepository(
            artifacts_dir=settings.artifacts_dir,
            r2_endpoint_url=settings.r2_endpoint_url,
            r2_access_key_id=settings.r2_access_key_id,
            r2_secret_access_key=settings.r2_secret_access_key,
            r2_bucket_name=settings.r2_bucket_name,
        )
        rows, excluded = await repo.get_player_pool(
            db,
            season_start=payload.season_start,
            min_qt_a=1,
            ruleset=payload.config.ruleset,
            return_exclusions=True,
        )
        n_excluded_no_projection = len(excluded)
        pool = [
            Player(
                player_id=r["player_id"],
                name=r["name"],
                role=cast(Role, r["role"]),
                real_team=r["real_team"],
                cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
                season_value=r.get("season_value"),
                start_probability=r.get("start_probability"),
                eligible_roles=frozenset(r.get("eligible_roles") or []),
                reliability_weight=r.get("reliability_weight"),
                sample_cohort=r.get("sample_cohort"),
            )
            for r in rows
        ]
        if not pool:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Empty player pool from DB for season_start="
                    f"{payload.season_start} (min_qt_a=1). Check that "
                    f"quotations and ML predictions are available, or "
                    f"pass an explicit player_pool."
                ),
            )

    session = AuctionSession(participants, auction_cfg, pool)
    session_id = uuid.uuid4().hex
    _get_session_store(request)[session_id] = session

    logger.info(
        "auction_session_initialized session_id=%s participants=%d pool=%d excluded_no_projection=%d",
        session_id,
        len(participants),
        len(pool),
        n_excluded_no_projection,
    )
    return InitializeAuctionResponse(
        session_id=session_id,
        n_excluded_no_projection=n_excluded_no_projection,
    )


@router.post(
    "/simulate",
    response_model=AuctionSimulationResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("member"))],
)
async def simulate_auction_endpoint(
    payload: SimulateAuctionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AuctionSimulationResponse:
    """Stateless Monte Carlo auction simulation. Does not touch auction_sessions."""
    inflation = _resolve_inflation(payload.config)
    auction_cfg = _auction_config_from_schema(payload.config, inflation)
    participants = [_participant_from_schema(p) for p in payload.participants]
    profiles = [_bidder_profile_from_schema(p) for p in payload.bidder_profiles]
    sim_cfg = _sim_config_from_schema(payload.sim_config)

    n_excluded_no_projection = 0
    if payload.player_pool is not None:
        pool: list[Player] = [_player_from_schema(p) for p in payload.player_pool]
    else:
        repo = DataRepository(
            artifacts_dir=settings.artifacts_dir,
            r2_endpoint_url=settings.r2_endpoint_url,
            r2_access_key_id=settings.r2_access_key_id,
            r2_secret_access_key=settings.r2_secret_access_key,
            r2_bucket_name=settings.r2_bucket_name,
        )
        rows, excluded = await repo.get_player_pool(
            db,
            season_start=payload.season_start,
            min_qt_a=1,
            ruleset=payload.config.ruleset,
            return_exclusions=True,
        )
        n_excluded_no_projection = len(excluded)
        pool = [
            Player(
                player_id=r["player_id"], name=r["name"], role=cast(Role, r["role"]),
                real_team=r["real_team"], cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
                season_value=r.get("season_value"), start_probability=r.get("start_probability"),
                eligible_roles=frozenset(r.get("eligible_roles") or []),
                reliability_weight=r.get("reliability_weight"),
                sample_cohort=r.get("sample_cohort"),
            )
            for r in rows
        ]
        if not pool:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty player pool")

    store_before = len(getattr(request.app.state, "auction_sessions", {}) or {})
    try:
        result = await asyncio.to_thread(
            simulate_auction, participants, profiles, auction_cfg, pool, sim_cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    store_after = len(getattr(request.app.state, "auction_sessions", {}) or {})
    if store_after != store_before:
        logger.error("simulate_auction mutated auction_sessions (%d -> %d)", store_before, store_after)

    return AuctionSimulationResponse(
        n_completed=result.n_completed,
        per_participant={
            pid: ParticipantSimStatsSchema(
                spend_p10=s.spend_p10, spend_p50=s.spend_p50, spend_p90=s.spend_p90,
                esv_total_p10=s.esv_total_p10, esv_total_p50=s.esv_total_p50, esv_total_p90=s.esv_total_p90,
                completion_probability=s.completion_probability,
                squad_composition_mode=dict(s.squad_composition_mode),
                top_players=[
                    {
                        "player_id": tp.player_id,
                        "name": tp.name,
                        "role": tp.role,
                        "frequency": tp.frequency,
                        "avg_price": tp.avg_price,
                    }
                    for tp in (s.top_players or ())
                ],
                typical_squad=[
                    {
                        "player_id": tp.player_id,
                        "name": tp.name,
                        "role": tp.role,
                        "frequency": tp.frequency,
                        "avg_price": tp.avg_price,
                    }
                    for tp in (s.typical_squad or ())
                ],
            )
            for pid, s in result.per_participant.items()
        },
        price_index_drift_p50=result.price_index_drift_p50,
        player_acquisition_probability={
            pid: PlayerAcquisitionStatsSchema(prob=st["prob"], avg_price=st["avg_price"])
            for pid, st in result.player_acquisition_probability.items()
        },
        wall_time_seconds=result.wall_time_seconds,
        warnings=list(result.warnings),
        n_excluded_no_projection=n_excluded_no_projection,
    )


@router.post(
    "/{session_id}/record",
    response_model=RecordAssignmentResponse,
    dependencies=[Depends(require_role("member"))],
)
def record_assignment_endpoint(
    session_id: str,
    payload: RecordAssignmentRequest,
    request: Request,
) -> RecordAssignmentResponse:
    """Registra un'assegnazione.  In caso di rifiuto, ritorna 200 con i
    campi ``rejection_code``/``rejection_reason`` valorizzati e
    ``success=False``; questo consente al client di gestire le 4 regole
    di validazione senza distinzione di status code."""
    session = _get_session(request, session_id)
    result = session.record(
        player_id=payload.player_id,
        winner_participant_id=payload.winner_participant_id,
        final_price=payload.final_price,
        assigned_slot=payload.assigned_slot,
    )

    if not result.success:
        return RecordAssignmentResponse(
            success=False,
            rejection_code=result.rejection_code,
            rejection_reason=result.rejection_reason,
        )

    last = result.updated_state.assignments[-1]
    return RecordAssignmentResponse(
        success=True,
        sequence_number=last.sequence_number,
        price_index_after=last.price_index_after,
    )


@router.post(
    "/{session_id}/undo",
    response_model=AuctionSummarySchema,
    dependencies=[Depends(require_role("member"))],
)
def undo_last_assignment_endpoint(
    session_id: str,
    request: Request,
) -> AuctionSummarySchema:
    """Annulla l'ultima assegnazione registrata.  Ritorna 409 se non c'è
    nulla da annullare (assignments vuoto)."""
    session = _get_session(request, session_id)
    if not session.state.assignments:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="no assignments to undo",
        )
    session.undo()
    return _summary_to_schema(session.summary())


@router.get(
    "/{session_id}/projection/{player_id}",
    response_model=ProjectionResponse,
    dependencies=[Depends(require_role("member"))],
)
def get_projection_endpoint(
    session_id: str,
    player_id: str,
    request: Request,
) -> ProjectionResponse:
    """Ritorna il prezzo atteso corrente per un giocatore ancora nel pool.

    Solleva 404 se ``player_id`` non esiste nel pool originale dell'asta.
    """
    session = _get_session(request, session_id)
    state = session.state
    target = next((p for p in state.available_pool if p.player_id == player_id), None)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"player_id {player_id!r} not found in available pool "
                "(potentially already assigned or never existed)"
            ),
        )

    expected = session.projection(player_id)
    percentile = float(state.role_percentile_map.get(player_id, 0.0))
    tier = classify_tier(percentile, state.config.market_drift_config)
    return ProjectionResponse(
        player_id=player_id,
        expected_price=float(expected),
        tier=tier,
    )


@router.get(
    "/{session_id}/alternatives/{player_id}",
    response_model=AlternativesResponse,
    dependencies=[Depends(require_role("member"))],
)
def get_alternatives_endpoint(
    session_id: str,
    player_id: str,
    request: Request,
    participant_id: str | None = None,
    strategy_name: str | None = None,
) -> AlternativesResponse:
    """Suggerisce low-cost, closest e (WS3) fronte Pareto + bid caps."""
    session = _get_session(request, session_id)
    try:
        suggestion = session.alternatives(
            target_player_id=player_id,
            participant_id=participant_id,
            strategy_name=strategy_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return AlternativesResponse(
        target_player_id=suggestion.target_player_id,
        low_cost_alternative=(
            _player_to_summary(suggestion.low_cost_alternative)
            if suggestion.low_cost_alternative is not None
            else None
        ),
        closest_alternative=(
            _player_to_summary(suggestion.closest_alternative)
            if suggestion.closest_alternative is not None
            else None
        ),
        reason_if_none=suggestion.reason_if_none,
        diversified_alternatives=[
            _player_to_summary(p) for p in (suggestion.diversified_alternatives or ())
        ],
        max_affordable_bid=suggestion.max_affordable_bid,
        strategy_price_cap=suggestion.strategy_price_cap,
    )


@router.get(
    "/{session_id}/summary",
    response_model=AuctionSummarySchema,
    dependencies=[Depends(require_role("member"))],
)
def get_summary_endpoint(
    session_id: str,
    request: Request,
) -> AuctionSummarySchema:
    """Snapshot read-only dello stato corrente dell'asta."""
    session = _get_session(request, session_id)
    return _summary_to_schema(session.summary())


@router.get(
    "/{session_id}/pool",
    response_model=list[AuctionPlayerSummarySchema],
    dependencies=[Depends(require_role("member"))],
)
def list_pool_endpoint(
    session_id: str,
    request: Request,
    q: str | None = None,
) -> list[AuctionPlayerSummarySchema]:
    """Ritorna il pool di giocatori disponibili per l'asta, opzionalmente
    filtrato per nome (substring case-insensitive).

    Caso d'uso primario: il client usa questo endpoint per popolare una
    dropdown/auto-completamento; una volta scelto il giocatore target,
    ne usa il ``playerId`` su ``/projection/{player_id}`` e
    ``/alternatives/{player_id}``.

    Il filtro è applicato solo al campo ``name``: ``?q=spin`` matcha
    sia ``Spinazzola`` che ``Spingardi``.  Nessun match su team/role.
    Query vuota o assente ⇒ ritorna l'intero pool.
    """
    session = _get_session(request, session_id)
    pool = session.state.available_pool
    if q is not None:
        q_lower = q.strip().lower()
        if q_lower:
            pool = [p for p in pool if q_lower in p.name.lower()]
    return [_player_to_summary(p) for p in pool]


@router.get(
    "/{session_id}/serialize",
    response_model=SerializedAuctionStateResponse,
    dependencies=[Depends(require_role("member"))],
)
def serialize_session_endpoint(
    session_id: str,
    request: Request,
) -> SerializedAuctionStateResponse:
    """Serializza lo stato corrente dell'asta per save/resume."""
    session = _get_session(request, session_id)
    payload = session.serialize()
    return SerializedAuctionStateResponse(payload=payload)


@router.post(
    "/deserialize",
    response_model=InitializeAuctionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("member"))],
)
def deserialize_session_endpoint(
    payload: dict[str, object],
    request: Request,
) -> InitializeAuctionResponse:
    """Ricostruisce una sessione d'asta da un payload serializzato."""
    try:
        restored_state = deserialize_state(payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid serialized payload: {exc}",
        ) from exc

    # Ricostruiamo un AuctionSession a partire dallo state deserializzato
    # mantenendo lo stesso contratto (init → session_id).
    # AuctionSession.__init__ chiama initialize_auction, quindi non possiamo
    # usare direttamente il constructor: creiamo un'istanza "shell" e
    # sostituiamo lo stato.
    placeholder = AuctionSession.__new__(AuctionSession)
    placeholder._state = restored_state  # noqa: SLF001 (intentional bypass)
    placeholder._pool = list(restored_state.available_pool)  # noqa: SLF001

    session_id = uuid.uuid4().hex
    _get_session_store(request)[session_id] = placeholder

    logger.info("auction_session_deserialized session_id=%s", session_id)
    return InitializeAuctionResponse(session_id=session_id)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_role("member"))],
)
def discard_session_endpoint(
    session_id: str,
    request: Request,
) -> None:
    """Termina e rimuove una sessione d'asta (libera memoria)."""
    store = _get_session_store(request)
    if session_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session_id {session_id!r} not found",
        )
    del store[session_id]
    logger.info("auction_session_discarded session_id=%s", session_id)


@router.get(
    "/{session_id}/var-ranking",
    response_model=VarRankingResponse,
    dependencies=[Depends(require_role("member"))],
    summary="VAR/ESV ranking of available players",
)
def get_var_ranking(
    session_id: str,
    request: Request,
) -> VarRankingResponse:
    """Returns available players ranked by Expected Surplus Value (ESV descending).

    Uses EWMA price projections from the live session when available
    (price_drift.get_current_projection), bypassing the uncalibrated DemandCurve.
    """
    store = _get_session_store(request)
    if session_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session_id {session_id!r} not found",
        )
    session: AuctionSession = store[session_id]
    state = session.state
    pool = list(state.available_pool)

    if not pool:
        return VarRankingResponse(session_id=session_id, items=[], using_live_prices=True)

    # Build EWMA price overrides for every available player
    price_overrides: dict[str, float] = {
        p.player_id: get_current_projection(state, p.player_id, pool)
        for p in pool
    }

    players_input = [
        {
            "player_id": p.player_id,
            "role": p.role,
            "projected_score": p.projected_score,
            "season_value": p.season_value,
            "start_probability": p.start_probability,
            "eligible_roles": sorted(p.eligible_roles) if p.eligible_roles else None,
            "fp_ibrido": p.fp_ibrido,
        }
        for p in pool
    ]

    valuation_mode = ValuationMode(
        getattr(state.config, "valuation_mode", "PER_MATCH_RATING")
    )
    engine = VarEngine(
        total_budget=state.config.budget_initial,
        roster_slots=dict(state.config.role_quotas),
        valuation_mode=valuation_mode,
        num_participants=state.config.num_participants,
        replacement_method=getattr(state.config, "replacement_method", "percentile"),
        min_start_probability=getattr(state.config, "min_start_probability", None),
        hybrid_blend=float(getattr(state.config, "hybrid_blend", 0.0) or 0.0),
    )
    results = engine.evaluate(players_input, price_overrides=price_overrides)

    player_map = {p.player_id: p for p in pool}
    items = [
        VarRankingItemSchema(
            player_id=r.player_id,
            name=player_map[r.player_id].name,
            role=r.role,
            projected_score=player_map[r.player_id].projected_score,
            var_score=r.var_score,
            expected_price=r.expected_price,
            esv=r.esv,
            calibrated=r.calibrated,
            buy_signal=r.esv > 0,
            season_value=player_map[r.player_id].season_value,
            start_probability=player_map[r.player_id].start_probability,
            sample_cohort=player_map[r.player_id].sample_cohort,
            reliability_weight=player_map[r.player_id].reliability_weight,
        )
        for r in results
    ]

    logger.info("var_ranking session_id=%s n=%d", session_id, len(items))
    return VarRankingResponse(session_id=session_id, items=items, using_live_prices=True)
