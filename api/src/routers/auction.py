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
    AuctionSummarySchema,
    InitializeAuctionRequest,
    InitializeAuctionResponse,
    MarketDriftConfigSchema,
    ProjectionResponse,
    RecordAssignmentRequest,
    RecordAssignmentResponse,
    SerializedAuctionStateResponse,
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
from ml.auction.var import VarEngine
from ml.optimizer.inflation import InflationConfig
from ml.optimizer.models import Player

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auction", tags=["auction"])


# ---------------------------------------------------------------------------
# Type-safe mappers: Pydantic ↔ internal dataclasses
# ---------------------------------------------------------------------------


def _player_from_schema(p: AuctionPlayerSchema) -> Player:
    return Player(
        player_id=p.player_id,
        name=p.name,
        role=cast(Role, p.role),
        real_team=p.real_team,
        cost=p.cost,
        projected_score=p.projected_score,
        season_value=p.season_value,
        start_probability=p.start_probability,
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
        market_drift_config=_market_drift_from_schema(cfg.market_drift_config),
        alternatives_config=_alternatives_config_from_schema(cfg.alternatives_config),
        use_inflation_baseline=cfg.use_inflation_baseline,
        inflation_config=inflation,
        valuation_mode=cfg.valuation_mode,
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
    )


def _price_index_to_dict(
    price_index: dict[Role, dict[Tier, float]],
) -> dict[str, dict[str, float]]:
    return {role: {tier: float(v) for tier, v in tiers.items()} for role, tiers in price_index.items()}


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
        # Il client non può passare un InflationConfig custom: l'inflazione
        # è deterministica lato server tramite default.  Se in futuro serve
        # esporla, si aggiunge uno schema dedicato.
        inflation = InflationConfig()

    auction_cfg = _auction_config_from_schema(payload.config, inflation)
    participants = [_participant_from_schema(p) for p in payload.participants]

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
        rows = await repo.get_player_pool(
            db,
            season_start=payload.season_start,
            min_qt_a=1,
        )
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
        "auction_session_initialized session_id=%s participants=%d pool=%d",
        session_id,
        len(participants),
        len(pool),
    )
    return InitializeAuctionResponse(session_id=session_id)


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
    summary = session.summary()
    return AuctionSummarySchema(
        participants=[_participant_to_schema(p) for p in summary.participants],
        assignments=[_assignment_to_schema(a) for a in summary.assignments],
        price_index=_price_index_to_dict(summary.price_index),
    )


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
) -> AlternativesResponse:
    """Suggerisce low-cost e closest match per il giocatore target."""
    session = _get_session(request, session_id)
    try:
        suggestion = session.alternatives(target_player_id=player_id)
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
    summary = session.summary()
    return AuctionSummarySchema(
        participants=[_participant_to_schema(p) for p in summary.participants],
        assignments=[_assignment_to_schema(a) for a in summary.assignments],
        price_index=_price_index_to_dict(summary.price_index),
    )


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
        )
        for r in results
    ]

    logger.info("var_ranking session_id=%s n=%d", session_id, len(items))
    return VarRankingResponse(session_id=session_id, items=items, using_live_prices=True)
