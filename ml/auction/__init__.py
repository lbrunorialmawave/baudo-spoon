"""Fantacalcio Live Auction Tracker & Dynamic Pricing Assistant.

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

__all__ = [
    # models
    "ADJACENT_TIERS",
    "ALL_TIERS",
    "AlternativesConfig",
    "AlternativeSuggestion",
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
    "initialize_auction",
    "record_assignment",
    "undo_last_assignment",
    "get_auction_summary",
    "serialize_state",
    "deserialize_state",
    # pure price drift
    "classify_tier",
    "build_initial_price_index",
    "compute_baseline_cost",
    "compute_expected_price",
    "get_current_projection",
    "project_price_for_player",
    "update_price_index",
    # pure alternatives
    "suggest_alternatives",
]
