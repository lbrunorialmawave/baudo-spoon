"""Single-matchday lineup optimizer (exact assignment over official modules)."""

from .optimizer import (
    LineupCandidate,
    SlotAssignment,
    FormationResult,
    OptimizeResult,
    optimize_lineup,
    DEFAULT_MIN_STARTER_PROB,
    compute_ev,
    opponent_adjustment,
)
from .enrichment import (
    MatchdayInfo,
    HybridInfo,
    EnrichmentStats,
    enrich_matched_players,
    parse_hybrid_rows,
    parse_matchday_rows,
)

__all__ = [
    "LineupCandidate",
    "SlotAssignment",
    "FormationResult",
    "OptimizeResult",
    "optimize_lineup",
    "DEFAULT_MIN_STARTER_PROB",
    "compute_ev",
    "opponent_adjustment",
    "MatchdayInfo",
    "HybridInfo",
    "EnrichmentStats",
    "enrich_matched_players",
    "parse_hybrid_rows",
    "parse_matchday_rows",
]
