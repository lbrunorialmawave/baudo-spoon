"""Stateful orchestrator for the live auction tracker.

This module is the *only* part of the package that mutates
:class:`AuctionState`.  It exposes the operator-facing operations:

* :func:`initialize_auction` - bootstrap a fresh state from a participant
  list and configuration.
* :func:`record_assignment` - validate and register one assignment with
  4-step explicit validation.
* :func:`undo_last_assignment` - revert the most recent assignment
  (common operator error: typo of price or winner).
* :func:`get_auction_summary` - read-only snapshot for periodic reporting.
* :func:`serialize_state` / :func:`deserialize_state` - structured
  (de)serialization for save/resume of an interrupted auction.

All state changes go through this orchestrator, so the price drift and
alternative-suggestion logic stays pure and unit-testable.
"""

from __future__ import annotations

import copy
import logging
from typing import cast

from ml.auction.alternatives import suggest_alternatives
from ml.auction.models import (
    AlternativeSuggestion,
    AlternativesConfig,
    AssignmentRecord,
    AuctionConfig,
    AuctionState,
    AuctionSummary,
    MarketDriftConfig,
    ParticipantSetup,
    ParticipantState,
    RecordResult,
    Role,
    Tier,
)
from ml.auction.price_drift import (
    build_initial_price_index,
    classify_tier,
    compute_expected_price,
    get_current_projection,
    update_price_index,
)
from ml.optimizer.inflation import InflationConfig, compute_role_percentile_map
from ml.optimizer.models import Player
from ml.optimizer.team_strength import load_team_strength_scores

logger = logging.getLogger(__name__)

__all__ = [
    "initialize_auction",
    "record_assignment",
    "undo_last_assignment",
    "get_auction_summary",
    "serialize_state",
    "deserialize_state",
    "AuctionSession",
    "suggest_alternatives",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_total_squad_size(role_quotas: dict[Role, int]) -> int:
    return sum(int(q) for q in role_quotas.values())


def _validate_inflation_config(
    config: AuctionConfig,
) -> None:
    """Verifica che l'oggetto ``inflation_config`` aderisca al protocollo
    atteso da :mod:`ml.optimizer.inflation`.  Il typing è lasco per
    disaccoppiare il modulo auction dai vincoli di ``InflationConfig``.
    """
    if not config.use_inflation_baseline:
        return
    cfg = config.inflation_config
    if cfg is None:
        raise ValueError(
            "AuctionConfig.inflation_config is required when "
            "use_inflation_baseline=True"
        )
    if not isinstance(cfg, InflationConfig):
        raise TypeError(
            "AuctionConfig.inflation_config must be an InflationConfig "
            f"instance, got {type(cfg).__name__}"
        )


def _as_dict(value: object) -> dict[object, object]:
    """Cast type-safe di un valore generico a ``dict``.

    Usato solo nei punti di deserializzazione dove il payload è tipato
    ``object`` per via di ``mypy --strict``; i cast puntuali sui valori
    sono già effettuati dal chiamante.
    """
    return cast(dict[object, object], value)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_auction(
    participants: list[ParticipantSetup],
    config: AuctionConfig,
    player_pool: list[Player],
) -> AuctionState:
    """Inizializza un'asta: valida setup, prepara stato, indicizza i ruoli.

    Parameters
    ----------
    participants:
        Lista di setup partecipanti.  Deve avere lunghezza pari a
        ``config.num_participants`` e ``participant_id`` unici.
    config:
        Configurazione di mercato e delle quote.
    player_pool:
        Pool completo di giocatori disponibili.  Viene copiato shallow
        per evitare mutazioni esterne allo stato dell'asta.

    Returns
    -------
    Un :class:`AuctionState` pronto per l'uso.

    Raises
    ------
    ValueError
        Se le validazioni di setup falliscono (numero partecipanti,
        unicità id, configurazione non coerente).
    """
    if len(participants) != config.num_participants:
        raise ValueError(
            f"len(participants)={len(participants)} does not match "
            f"config.num_participants={config.num_participants}"
        )

    seen_ids: set[str] = set()
    for p in participants:
        if p.participant_id in seen_ids:
            raise ValueError(
                f"duplicate participant_id: {p.participant_id!r}"
            )
        seen_ids.add(p.participant_id)

    _validate_inflation_config(config)

    # Calcola la mappa percentile una volta sola, sul pool completo.
    percentile_map = compute_role_percentile_map(player_pool)

    participants_state: dict[str, ParticipantState] = {
        p.participant_id: ParticipantState(
            participant_id=p.participant_id,
            display_name=p.display_name,
            budget_residual=p.budget_initial,
            squad=[],
            role_breakdown={role: 0 for role in config.role_quotas},
        )
        for p in participants
    }

    # Load team-strength Elo scores once per session for inflation adjustment.
    ts_scores: dict[str, float] = {}
    inflation_cfg = config.inflation_config
    if (
        config.use_inflation_baseline
        and inflation_cfg is not None
        and getattr(inflation_cfg, "team_strength_multiplier", 0.0) > 0
    ):
        known_teams = {p.real_team for p in player_pool if p.real_team}
        ts_scores = load_team_strength_scores(known_teams=known_teams)

    state = AuctionState(
        config=config,
        participants=participants_state,
        assignments=[],
        price_index=build_initial_price_index(config),
        available_pool=list(player_pool),
        role_percentile_map=percentile_map,
        team_strength_scores=ts_scores,
    )

    logger.info(
        "auction_initialized participants=%d pool_size=%d",
        len(participants),
        len(player_pool),
    )
    return state


# ---------------------------------------------------------------------------
# Record assignment
# ---------------------------------------------------------------------------


def _resolve_assigned_slot(
    state: AuctionState,
    player: Player,
    winner: ParticipantState,
    assigned_slot: str | None,
) -> tuple[str | None, str | None]:
    """Decide lo slot di ruolo da riempire e valida la quota residua.

    Returns
    -------
    (slot, rejection_code_or_None)
        Se ``rejection_code`` è non-None, ``slot`` è None e il chiamante
        deve rifiutare l'assegnazione.
    """
    quotas = state.config.role_quotas
    ruleset = state.config.ruleset

    if ruleset == "CLASSIC":
        role = player.role
        if assigned_slot is not None and assigned_slot != role:
            return None, "invalid_slot"
        quota = quotas.get(role, 0)
        filled = winner.role_breakdown.get(role, 0)
        if filled >= quota:
            return None, "role_full"
        return role, None

    # MANTRA: slot must be one of the player's eligible_roles (or a single
    # explicit role if eligible_roles is empty — defensive fallback).
    eligible = set(player.eligible_roles) if player.eligible_roles else set()
    if not eligible:
        eligible = {player.role}

    if assigned_slot is not None:
        if assigned_slot not in eligible:
            return None, "invalid_slot"
        if assigned_slot not in quotas:
            return None, "invalid_slot"
        filled = winner.role_breakdown.get(assigned_slot, 0)
        if filled >= quotas[assigned_slot]:
            return None, "role_full"
        return assigned_slot, None

    # Auto-pick: among eligible slots that still have residual quota for
    # the winner, prefer the one with the *least residual capacity*
    # (scarcest remaining slot for this participant). Ties broken by
    # stable role-name order for determinism.
    candidates = [
        r
        for r in eligible
        if r in quotas and winner.role_breakdown.get(r, 0) < quotas[r]
    ]
    if not candidates:
        return None, "role_full"

    def _scarcity_key(r: str) -> tuple[int, str]:
        residual = quotas[r] - winner.role_breakdown.get(r, 0)
        return (residual, r)

    return min(candidates, key=_scarcity_key), None


def record_assignment(
    state: AuctionState,
    player_id: str,
    winner_participant_id: str,
    final_price: int,
    assigned_slot: str | None = None,
) -> RecordResult:
    """Valida e registra un'assegnazione, aggiornando lo stato in place.

    Le validazioni sono eseguite in ordine; al primo errore lo stato non
    viene mutato e viene restituito un :class:`RecordResult` con
    ``success=False`` e ``rejection_reason`` esplicito.

    Parameters
    ----------
    assigned_slot:
        Slot di ruolo MANTRA effettivamente occupato (es. ``"Dd"``).
        Opzionale: se omesso in modalità MANTRA, l'orchestratore sceglie
        automaticamente tra gli ``eligible_roles`` del giocatore lo slot
        con minor capacità residua per il vincitore. In CLASSIC viene
        ignorato (si usa sempre ``player.role``).
    """
    # -- 1. player_id esiste nel pool originale (lookup in available_pool
    #       è sufficiente perché available_pool è la fonte di verità
    #       per i giocatori non ancora assegnati).
    player = _find_player(state.available_pool, player_id)
    if player is None:
        # Distingui "mai esistito" da "già assegnato" per dare un errore
        # diagnostico più utile.
        already = any(
            r.player.player_id == player_id for r in state.assignments
        )
        if already:
            return _reject(
                "player_already_assigned",
                f"giocatore {player_id!r} gia' assegnato in un'asta precedente",
            )
        return _reject(
            "unknown_player",
            f"player_id {player_id!r} non presente nel pool",
        )

    # -- 2. winner_participant_id esiste.
    winner = state.participants.get(winner_participant_id)
    if winner is None:
        return _reject(
            "unknown_participant",
            f"winner_participant_id {winner_participant_id!r} non esiste",
        )

    # -- 3. risoluzione e validazione dello slot di ruolo.
    slot, slot_err = _resolve_assigned_slot(
        state, player, winner, assigned_slot
    )
    if slot_err == "invalid_slot":
        return _reject(
            "invalid_slot",
            (
                f"slot {assigned_slot!r} non valido per il giocatore "
                f"{player_id!r} (eligible={sorted(player.eligible_roles) if player.eligible_roles else [player.role]})"
            ),
        )
    if slot_err == "role_full" or slot is None:
        role_label = assigned_slot or player.role
        quota = state.config.role_quotas.get(role_label, 0)
        filled = winner.role_breakdown.get(role_label, 0)
        return _reject(
            "role_full",
            (
                f"partecipante {winner.participant_id!r} ha gia' completato "
                f"il ruolo {role_label} ({filled}/{quota})"
            ),
        )

    # ``role`` sullo AssignmentRecord resta il codice usato per price index
    # e breakdown: in CLASSIC = player.role; in MANTRA = assigned_slot.
    role = slot

    # -- 4. regola della riserva crediti:
    #       final_price <= budget_residual - (slots_ancora_da_riempire - 1)
    total_squad_size = _compute_total_squad_size(state.config.role_quotas)
    current_squad_size = sum(winner.role_breakdown.values())
    slots_remaining_before = total_squad_size - current_squad_size
    # L'acquisto attuale riempie uno slot; dopo, devono restare crediti >= 1
    # per ciascuno degli slot ancora vuoti (slots_remaining_before - 1).
    max_allowed = winner.budget_residual - (slots_remaining_before - 1)
    if final_price > max_allowed:
        return _reject(
            "credit_reserve_violation",
            (
                f"prezzo {final_price} viola la regola della riserva crediti: "
                f"max consentito {max_allowed} "
                f"(budget_residuo={winner.budget_residual}, "
                f"slot_ancora_da_riempire={slots_remaining_before})"
            ),
        )

    if final_price < 0:
        return _reject(
            "negative_price",
            f"final_price deve essere >= 0, ricevuto {final_price}",
        )

    # -- Calcola tier, expected price ed applica EWMA PRIMA di mutare lo stato.
    # Lo slot assegnato guida l'aggiornamento del price_index (Fase 3:
    # non più la risoluzione speculativa di resolve_pricing_role).
    percentile = state.role_percentile_map.get(player_id, 0.0)
    tier = classify_tier(percentile, state.config.market_drift_config)
    expected_price = compute_expected_price(
        player=player,
        role_percentile=percentile,
        role=role,
        tier=tier,
        price_index=state.price_index,
        config=state.config,
        team_strength_scores=state.team_strength_scores or None,
    )

    index_before, index_after, snapshot_before = update_price_index(
        role=role,
        tier=tier,
        actual_price=float(final_price),
        expected_price=expected_price,
        price_index=state.price_index,
        config=state.config.market_drift_config,
    )

    # -- Mutazioni dello stato (post-validazione).
    winner.budget_residual -= final_price
    winner.squad.append(player)
    winner.role_breakdown[role] = winner.role_breakdown.get(role, 0) + 1
    # Rimuovi il giocatore dal pool disponibile.
    state.available_pool = [p for p in state.available_pool if p.player_id != player_id]

    seq = len(state.assignments) + 1
    record = AssignmentRecord(
        sequence_number=seq,
        player=player,
        winner_participant_id=winner_participant_id,
        final_price=final_price,
        role=role,  # type: ignore[arg-type]
        tier=tier,
        price_index_before=index_before,
        price_index_after=index_after,
        price_index_snapshot_before=snapshot_before,
        assigned_slot=role,
    )
    state.assignments.append(record)

    logger.info(
        "assignment_recorded seq=%d player=%s winner=%s role=%s slot=%s "
        "tier=%s price=%d expected=%.2f index=%.4f->%.4f budget_residuo=%d",
        seq,
        player_id,
        winner_participant_id,
        role,
        role,
        tier,
        final_price,
        expected_price,
        index_before,
        index_after,
        winner.budget_residual,
    )

    return RecordResult(success=True, updated_state=state)


def _reject(code: str, reason: str) -> RecordResult:
    logger.warning("assignment_rejected code=%s reason=%s", code, reason)
    return RecordResult(
        success=False,
        updated_state=None,
        rejection_reason=reason,
        rejection_code=code,
    )


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def undo_last_assignment(
    state: AuctionState,
) -> AuctionState:
    """Annulla l'ultima assegnazione registrata, ripristinando lo stato.

    Solleva :class:`IndexError` se non ci sono assegnazioni da annullare
    (l'operatore non dovrebbe mai trovarsi in questo stato, ma la difesa
    esplicita è meglio di un'eccezione generica).
    """
    if not state.assignments:
        raise IndexError("nessuna assegnazione da annullare")

    last = state.assignments[-1]
    winner = state.participants[last.winner_participant_id]
    if winner is None:  # pragma: no cover - non dovrebbe accadere
        raise ValueError(
            f"winner {last.winner_participant_id!r} inconsistente con lo stato"
        )

    # 1) Ripristina budget, roster e breakdown.
    # Prefer assigned_slot (MANTRA multi-role) with fallback to role for
    # legacy records deserialized without the field.
    slot = last.assigned_slot if last.assigned_slot is not None else last.role
    winner.budget_residual += last.final_price
    if winner.squad and winner.squad[-1].player_id == last.player.player_id:
        winner.squad.pop()
    else:
        # Squad non in coda: rimuovi comunque (idempotente sulla presenza).
        winner.squad = [
            p for p in winner.squad if p.player_id != last.player.player_id
        ]
    if winner.role_breakdown.get(slot, 0) > 0:
        winner.role_breakdown[slot] -= 1

    # 2) Ripristina il price_index dallo snapshot deterministico.
    if last.price_index_snapshot_before:
        state.price_index = copy.deepcopy(last.price_index_snapshot_before)
    else:
        # Fallback difensivo: nessuno snapshot presente, ripristina a 1.0
        # per tutte le combinazioni (non ideale ma evita corruzione).
        logger.warning(
            "undo_fallback_no_snapshot seq=%d player=%s",
            last.sequence_number,
            last.player.player_id,
        )
        state.price_index = build_initial_price_index(state.config)

    # 3) Riposiziona il giocatore nel pool disponibile, preservando
    #    l'ordinamento originale.
    state.available_pool.append(last.player)

    # 4) Rimuovi il record.
    state.assignments.pop()

    logger.info(
        "assignment_undone seq=%d player=%s winner=%s",
        last.sequence_number,
        last.player.player_id,
        last.winner_participant_id,
    )
    return state


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_auction_summary(
    state: AuctionState,
    *,
    include_completion_probability: bool = True,
) -> AuctionSummary:
    """Restituisce un riepilogo immutabile dello stato corrente.

    When ``include_completion_probability`` is True (default), attaches
    the WS3 #1 live indicator ``completion_probability`` per participant.

    When ``ruleset=MANTRA``, also attaches per-participant
    ``mantra_module_coverage`` (informational residual fieldability of
    the 11 official Mantra Experience modules).
    """
    completion: dict[str, float] | None = None
    if include_completion_probability:
        try:
            from ml.auction.completion_probability import (
                estimate_all_completion_probabilities,
            )

            completion = estimate_all_completion_probabilities(state)
        except Exception:  # pragma: no cover - defensive
            logger.exception("completion_probability_failed")
            completion = None

    mantra_cov: dict[str, dict[str, object]] | None = None
    if state.config.ruleset == "MANTRA":
        try:
            from ml.optimizer.formations import evaluate_all_coverages

            mantra_cov = {
                pid: dict(evaluate_all_coverages(pstate.squad))
                for pid, pstate in state.participants.items()
            }
        except Exception:  # pragma: no cover - defensive
            logger.exception("mantra_module_coverage_failed")
            mantra_cov = None

    return AuctionSummary(
        participants=list(state.participants.values()),
        assignments=list(state.assignments),
        price_index=copy.deepcopy(state.price_index),
        completion_probability=completion,
        mantra_module_coverage=mantra_cov,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_state(state: AuctionState) -> dict[str, object]:
    """Serializza lo stato in un dizionario JSON-compatibile.

    Il payload può essere persistito con ``json.dump``; per ricaricarlo
    usare :func:`deserialize_state`.  I ``Player`` sono serializzati
    come ``dataclasses.asdict``; nessun riferimento a risorse esterne.
    """
    return {
        "config": _config_to_dict(state.config),
        "participants": {
            pid: {
                "participant_id": ps.participant_id,
                "display_name": ps.display_name,
                "budget_residual": ps.budget_residual,
                "squad": [_player_to_dict(p) for p in ps.squad],
                "role_breakdown": dict(ps.role_breakdown),
            }
            for pid, ps in state.participants.items()
        },
        "assignments": [
            {
                "sequence_number": r.sequence_number,
                "player": _player_to_dict(r.player),
                "winner_participant_id": r.winner_participant_id,
                "final_price": r.final_price,
                "role": r.role,
                "tier": r.tier,
                "price_index_before": r.price_index_before,
                "price_index_after": r.price_index_after,
                "price_index_snapshot_before": {
                    role: dict(tiers)
                    for role, tiers in r.price_index_snapshot_before.items()
                },
                "assigned_slot": r.assigned_slot if r.assigned_slot is not None else r.role,
            }
            for r in state.assignments
        ],
        "price_index": {
            role: dict(tiers) for role, tiers in state.price_index.items()
        },
        "available_pool": [_player_to_dict(p) for p in state.available_pool],
        "role_percentile_map": dict(state.role_percentile_map),
        # Elo scores are loaded once per session in ``initialize_auction``
        # and are needed by every ``record_assignment`` that prices a
        # player.  Without round-tripping them, a resumed auction would
        # silently lose the team-strength adjustment (the field has a
        # default_factory=dict, so no error is raised — the Elo signal
        # just disappears).
        "team_strength_scores": dict(state.team_strength_scores),
    }


def deserialize_state(payload: dict[str, object]) -> AuctionState:
    """Ricostruisce uno :class:`AuctionState` da un payload serializzato."""
    config = _config_from_dict(cast(dict[str, object], payload["config"]))
    players_by_id: dict[str, Player] = {}

    raw_participants = cast(
        dict[str, dict[str, object]], payload["participants"]
    )
    participants: dict[str, ParticipantState] = {}
    for pid, ps in raw_participants.items():
        squad = [
            _player_from_dict(cast(dict[str, object], d), players_by_id)
            for d in cast(list[object], ps["squad"])
        ]
        participants[pid] = ParticipantState(
            participant_id=cast(str, ps["participant_id"]),
            display_name=cast(str, ps["display_name"]),
            budget_residual=cast(int, ps["budget_residual"]),
            squad=squad,
            role_breakdown=cast(
                dict[Role, int], _as_dict(ps["role_breakdown"])
            ),
        )

    raw_assignments = cast(list[dict[str, object]], payload["assignments"])
    assignments: list[AssignmentRecord] = []
    for r in raw_assignments:
        player = _player_from_dict(cast(dict[str, object], r["player"]), players_by_id)
        snapshot = {
            role: dict(tiers)
            for role, tiers in cast(
                dict[Role, dict[Tier, float]], r["price_index_snapshot_before"]
            ).items()
        }
        role_val = cast(Role, r["role"])
        # Backward compat: pre-Fase-3 payloads lack assigned_slot.
        assigned_slot_val = cast(str | None, r.get("assigned_slot", role_val))
        assignments.append(
            AssignmentRecord(
                sequence_number=cast(int, r["sequence_number"]),
                player=player,
                winner_participant_id=cast(str, r["winner_participant_id"]),
                final_price=cast(int, r["final_price"]),
                role=role_val,
                tier=cast(Tier, r["tier"]),
                price_index_before=cast(float, r["price_index_before"]),
                price_index_after=cast(float, r["price_index_after"]),
                price_index_snapshot_before=snapshot,
                assigned_slot=assigned_slot_val,
            )
        )

    price_index: dict[Role, dict[Tier, float]] = {
        role: dict(tiers)
        for role, tiers in cast(
            dict[Role, dict[Tier, float]], payload["price_index"]
        ).items()
    }
    available_pool = [
        _player_from_dict(cast(dict[str, object], d), players_by_id)
        for d in cast(list[object], payload["available_pool"])
    ]

    return AuctionState(
        config=config,
        participants=participants,
        assignments=assignments,
        price_index=price_index,
        available_pool=available_pool,
        role_percentile_map=cast(
            dict[str, float], _as_dict(payload["role_percentile_map"])
        ),
        # Restore the Elo table so a resumed auction keeps pricing
        # players with the team-strength adjustment.  Missing key
        # tolerated for backward compat with pre-fix payloads: fall
        # back to empty dict (no Elo signal — same as a brand-new
        # state with ``team_strength_multiplier=0.0``).
        team_strength_scores=cast(
            dict[str, float],
            _as_dict(payload.get("team_strength_scores", {})),
        ),
    )


# ---------------------------------------------------------------------------
# Convenience façade
# ---------------------------------------------------------------------------


class AuctionSession:
    """Facciata stateful che incapsula ``AuctionState`` e l'operatore.

    Espone i comandi ad alto livello pensati per essere invocati
    dall'operatore uno dopo l'altro, in modo lineare e sincrono
    (single-operator, single-process, nessun async, nessun network).
    """

    def __init__(
        self,
        participants: list[ParticipantSetup],
        config: AuctionConfig,
        player_pool: list[Player],
    ) -> None:
        self._state = initialize_auction(participants, config, player_pool)
        self._pool = list(player_pool)

    @property
    def state(self) -> AuctionState:
        return self._state

    def record(
        self,
        player_id: str,
        winner_participant_id: str,
        final_price: int,
        assigned_slot: str | None = None,
    ) -> RecordResult:
        return record_assignment(
            self._state,
            player_id,
            winner_participant_id,
            final_price,
            assigned_slot=assigned_slot,
        )

    def undo(self) -> AuctionState:
        return undo_last_assignment(self._state)

    def summary(self) -> AuctionSummary:
        return get_auction_summary(self._state)

    def projection(self, player_id: str) -> float:
        return get_current_projection(self._state, player_id, self._pool)

    def alternatives(
        self,
        target_player_id: str,
        config: AlternativesConfig | None = None,
        *,
        participant_id: str | None = None,
        strategy_name: str | None = None,
        diversify: bool = True,
    ) -> AlternativeSuggestion:
        cfg = config or self._state.config.alternatives_config
        target = next(
            (p for p in self._pool if p.player_id == target_player_id),
            None,
        )
        if target is None:
            raise ValueError(
                f"target_player_id {target_player_id!r} non presente "
                f"nel pool originale dell'asta"
            )
        from ml.auction.models import ValuationMode

        vm = ValuationMode(
            getattr(self._state.config, "valuation_mode", "PER_MATCH_RATING")
        )
        return suggest_alternatives(
            target=target,
            available_pool=self._state.available_pool,
            state=self._state,
            config=cfg,
            valuation_mode=vm,
            participant_id=participant_id,
            strategy_name=strategy_name,
            diversify=diversify,
        )

    def serialize(self) -> dict[str, object]:
        return serialize_state(self._state)


# ---------------------------------------------------------------------------
# Internal helpers (serialization)
# ---------------------------------------------------------------------------


def _player_to_dict(p: Player) -> dict[str, object]:
    out: dict[str, object] = {
        "player_id": p.player_id,
        "name": p.name,
        "real_team": p.real_team,
        "role": p.role,
        "cost": p.cost,
        "projected_score": p.projected_score,
    }
    if p.eligible_roles:
        out["eligible_roles"] = sorted(p.eligible_roles)
    if p.season_value is not None:
        out["season_value"] = p.season_value
    return out


def _player_from_dict(
    d: dict[str, object], cache: dict[str, Player]
) -> Player:
    pid = cast(str, d["player_id"])
    if pid in cache:
        return cache[pid]
    raw_eligible = d.get("eligible_roles")
    eligible: frozenset[str] = frozenset()
    if isinstance(raw_eligible, (list, tuple, set, frozenset)):
        eligible = frozenset(str(r) for r in raw_eligible)
    kwargs: dict[str, object] = {
        "player_id": pid,
        "name": cast(str, d["name"]),
        "real_team": cast(str, d["real_team"]),
        "role": cast(Role, d["role"]),
        "cost": cast(int, d["cost"]),
        "projected_score": cast(float, d["projected_score"]),
        "eligible_roles": eligible,
    }
    if "season_value" in d and d["season_value"] is not None:
        kwargs["season_value"] = cast(float, d["season_value"])
    p = Player(**kwargs)  # type: ignore[arg-type]
    cache[pid] = p
    return p


def _config_to_dict(config: AuctionConfig) -> dict[str, object]:
    return {
        "num_participants": config.num_participants,
        "role_quotas": dict(config.role_quotas),
        "ruleset": config.ruleset,
        "market_drift_config": {
            "alpha": config.market_drift_config.alpha,
            "spillover_adjacent_tier": config.market_drift_config.spillover_adjacent_tier,
            "spillover_cross_role": config.market_drift_config.spillover_cross_role,
            "min_index": config.market_drift_config.min_index,
            "max_index": config.market_drift_config.max_index,
            "tier_thresholds": list(config.market_drift_config.tier_thresholds),
        },
        "alternatives_config": {
            "low_cost_percentile": config.alternatives_config.low_cost_percentile,
        },
        "use_inflation_baseline": config.use_inflation_baseline,
        "reference_budget": config.reference_budget,
        "budget_initial": config.budget_initial,
        "valuation_mode": config.valuation_mode,
        "hybrid_blend": config.hybrid_blend,
        "risk_aversion": config.risk_aversion,
        "apply_reliability_weight": config.apply_reliability_weight,
        # inflation_config non è serializzato di default: il caller può
        # ricostruirlo se davvero necessario (l'ottimizzatore rosa lo
        # ottiene via DI).  Per completezza includiamo il repr ma non
        # lo roundtrippiamo.
    }


def _config_from_dict(d: dict[str, object]) -> AuctionConfig:
    md = _as_dict(d["market_drift_config"])
    market_drift_config = MarketDriftConfig(
        alpha=cast(float, md["alpha"]),
        spillover_adjacent_tier=cast(float, md["spillover_adjacent_tier"]),
        spillover_cross_role=cast(float, md["spillover_cross_role"]),
        min_index=cast(float, md["min_index"]),
        max_index=cast(float, md["max_index"]),
        tier_thresholds=cast(
            tuple[float, float],
            tuple(cast(list[float], md["tier_thresholds"])),
        ),
    )
    alt = _as_dict(d["alternatives_config"])
    alternatives_config = AlternativesConfig(
        low_cost_percentile=cast(float, alt["low_cost_percentile"]),
    )
    # Backward compat: pre-Fase-3 payloads lack ruleset → CLASSIC.
    ruleset = cast(str, d.get("ruleset", "CLASSIC"))
    return AuctionConfig(
        num_participants=cast(int, d["num_participants"]),
        role_quotas=cast(
            dict[Role, int],
            dict(cast(dict[object, object], d["role_quotas"])),
        ),
        ruleset=ruleset,  # type: ignore[arg-type]
        market_drift_config=market_drift_config,
        alternatives_config=alternatives_config,
        use_inflation_baseline=cast(
            bool, d.get("use_inflation_baseline", False)
        ),
        inflation_config=None,
        reference_budget=cast(int, d.get("reference_budget", 300)),
        budget_initial=cast(int, d.get("budget_initial", 300)),
        valuation_mode=cast(str, d.get("valuation_mode", "PER_MATCH_RATING")),
        hybrid_blend=float(d.get("hybrid_blend", 0.0) or 0.0),
        risk_aversion=float(d.get("risk_aversion", 0.0) or 0.0),
        apply_reliability_weight=bool(d.get("apply_reliability_weight", True)),
    )


# ---------------------------------------------------------------------------
# Internal helpers (player lookup)
# ---------------------------------------------------------------------------


def _find_player(pool: list[Player], player_id: str) -> Player | None:
    for p in pool:
        if p.player_id == player_id:
            return p
    return None
