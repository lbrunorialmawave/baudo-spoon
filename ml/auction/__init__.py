"""Live auction tracker — domain-level orchestration for real-time asta assignments.

Fantacalcio Live Auction Tracker & Dynamic Pricing Assistant.

Modulo a singolo operatore, single-process: registra in tempo reale le
assegnazioni di un'asta live e fornisce supporto decisionale (proiezione
dinamica dei prezzi + suggerimenti di alternative).

Architettura:

* :mod:`ml.auction.models` - dataclass immutabili di configurazione e
  dataclass mutabili di stato.
* :mod:`ml.auction.price_drift` - funzioni pure per l'aggiornamento
  EWMA del price index e la classificazione per tier.
* :mod:`ml.auction.alternatives` - funzione pura per i suggerimenti
  di alternative (low-cost + closest match).
* :mod:`ml.auction.orchestrator` - unica parte stateful: inizializza
  l'asta, valida e registra le assegnazioni, gestisce undo, serializza.

Il :class:`Player` e le costanti di ruolo/quotas sono **riusati** da
:mod:`ml.optimizer.models`.

INTEGRATION BOUNDARY (ml rearchitecture):
  This module consumes Player.cost and Player.projected_score from
  ml.optimizer.models.Player (PlayerV1 schema). It does NOT define
  baseline_cost as a field on Player.

  baseline_cost is a derived value computed in:
      ml.auction.price_drift.compute_baseline_cost(player, role_percentile, config)
  Formula: player.cost * (budget_initial / reference_budget) [* inflation if enabled]

  The ML rearchitecture (ml.domain, Phase 5) will produce PlayerV2.expected_auction_price
  which feeds into compute_baseline_cost as the static pre-auction prior.
  It does NOT replace or duplicate the dynamic EWMA price drift model.
  The boundary is: PlayerV2.expected_auction_price → baseline_cost (input) → price_drift
  (dynamic adjustment during the live auction).

PRICE MODEL BOUNDARY (three co-existing models):

  1. ml.optimizer.inflation.estimate_effective_cost
     Used for: pre-auction squad cost estimation in the optimizer.
     Input: Player.cost, role_percentile, InflationConfig.
     NOT used during a live auction.

  2. ml.auction.price_drift (EWMA)
     Used for: live session price projections, updated after every assignment.
     Entry points: get_current_projection(), compute_expected_price().
     This is the authoritative expected price during an active session.

  3. ml.auction.var.DemandCurve
     Used for: pre-session ESV/VAR ranking when no auction history exists.
     During a live session, bypass with override_expected_price from (2).
     Entry: VarEngine.evaluate(players, price_overrides={id: price, ...}).

  Consumers (e.g. bookmaker signals from Phase 6) should enter at (2) as an
  additional input to compute_baseline_cost, not by replacing the EWMA layer.
"""

from __future__ import annotations

from ml.auction.alternatives import suggest_alternatives
from ml.auction.models import (
    ADJACENT_TIERS,
    ALL_TIERS,
    AlternativesConfig,
    AlternativeSuggestion,
    AssignmentRecord,
    AuctionConfig,
    AuctionState,
    AuctionSummary,
    MarketDriftConfig,
    ParticipantSetup,
    ParticipantState,
    RecordResult,
)
from ml.auction.orchestrator import (
    AuctionSession,
    deserialize_state,
    get_auction_summary,
    initialize_auction,
    record_assignment,
    serialize_state,
    undo_last_assignment,
)
from ml.auction.price_drift import (
    build_initial_price_index,
    classify_tier,
    compute_baseline_cost,
    compute_expected_price,
    get_current_projection,
    project_price_for_player,
    update_price_index,
)

__all__ = [  # noqa: RUF022 — grouped by module/purpose, not alphabetically, on purpose
    # models
    "ADJACENT_TIERS",
    "ALL_TIERS",
    "AlternativeSuggestion",
    "AlternativesConfig",
    "AssignmentRecord",
    "AuctionConfig",
    "AuctionState",
    "AuctionSummary",
    "MarketDriftConfig",
    "ParticipantSetup",
    "ParticipantState",
    "RecordResult",
    # orchestrator
    "AuctionSession",
    "deserialize_state",
    "get_auction_summary",
    "initialize_auction",
    "record_assignment",
    "serialize_state",
    "undo_last_assignment",
    # pure price drift
    "build_initial_price_index",
    "classify_tier",
    "compute_baseline_cost",
    "compute_expected_price",
    "get_current_projection",
    "project_price_for_player",
    "update_price_index",
    # pure alternatives
    "suggest_alternatives",
]
