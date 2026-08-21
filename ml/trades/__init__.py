"""Trade advisor + fairness engine: coverage, retention, bilateral evaluation."""

from .credit_penalty import recompute_value_on_transfer, round_half_up
from .advisor import (
    TradeOutCandidate,
    TradeInTarget,
    CoverageCell,
    TradeDashboard,
    build_trade_dashboard,
    retention_score,
)
from .fairness import (
    EnrichedTradePlayer,
    PTVWeights,
    TradeEvaluation,
    evaluate_trade,
    player_trade_value,
)
from .signals import (
    FormaResult,
    MatchdayVote,
    TitolaritaResult,
    forma_recente_score,
    indice_titolarita,
)

__all__ = [
    "recompute_value_on_transfer",
    "round_half_up",
    "TradeOutCandidate",
    "TradeInTarget",
    "CoverageCell",
    "TradeDashboard",
    "build_trade_dashboard",
    "retention_score",
    "EnrichedTradePlayer",
    "PTVWeights",
    "TradeEvaluation",
    "evaluate_trade",
    "player_trade_value",
    "FormaResult",
    "MatchdayVote",
    "TitolaritaResult",
    "forma_recente_score",
    "indice_titolarita",
]
