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
    filter_votes_pre_match,
    blend_fp_with_form,
    form_blend_weight,
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
    "filter_votes_pre_match",
    "blend_fp_with_form",
    "form_blend_weight",
]
