"""Trade advisor: coverage gaps, retention score, credit penalty."""

from .credit_penalty import recompute_value_on_transfer, round_half_up
from .advisor import (
    TradeOutCandidate,
    TradeInTarget,
    CoverageCell,
    TradeDashboard,
    build_trade_dashboard,
    retention_score,
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
]
