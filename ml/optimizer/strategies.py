"""Default 4 strategy profiles (configurable, not hardcoded in the solver)."""

from __future__ import annotations

from typing import Final

from ml.optimizer.models import StrategyName, StrategyProfile

__all__ = ["DEFAULT_FOUR_STRATEGIES", "default_strategies"]

#: Default top-tier cost threshold for the MIXED strategy (credits).
DEFAULT_TOP_TIER_COST_THRESHOLD: Final[int] = 30
#: Default maximum number of top-tier players for the MIXED strategy.
DEFAULT_MAX_TOP_TIER_PLAYERS: Final[int] = 5

#: Default super-defensive share of budget for (P + D).
DEFAULT_SUPER_DEFENSIVE_SHARE: Final[float] = 0.45
#: Default super-offensive share of budget for (C + A).
DEFAULT_SUPER_OFFENSIVE_SHARE: Final[float] = 0.65


def _balanced() -> StrategyProfile:
    return StrategyProfile(
        name="BALANCED", role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0}
    )


def _super_defensive(share: float = DEFAULT_SUPER_DEFENSIVE_SHARE) -> StrategyProfile:
    return StrategyProfile(
        name="SUPER_DEFENSIVE",
        role_weight={"P": 1.2, "D": 1.3, "C": 1.0, "A": 0.8},
        min_budget_share_by_roles=(frozenset({"P", "D"}), share),
    )


def _super_offensive(share: float = DEFAULT_SUPER_OFFENSIVE_SHARE) -> StrategyProfile:
    return StrategyProfile(
        name="SUPER_OFFENSIVE",
        role_weight={"P": 0.8, "D": 0.9, "C": 1.15, "A": 1.3},
        min_budget_share_by_roles=(frozenset({"C", "A"}), share),
    )


def _mixed(
    threshold: int = DEFAULT_TOP_TIER_COST_THRESHOLD,
    max_top: int = DEFAULT_MAX_TOP_TIER_PLAYERS,
) -> StrategyProfile:
    return StrategyProfile(
        name="MIXED",
        role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
        max_top_tier_players=max_top,
        top_tier_cost_threshold=float(threshold),
    )


def default_strategies() -> tuple[StrategyProfile, ...]:
    """Restituisce la tupla delle 4 strategie di default.

    L'ordine è stabile: ``BALANCED, SUPER_DEFENSIVE, SUPER_OFFENSIVE, MIXED``.
    """
    return (
        _balanced(),
        _super_defensive(),
        _super_offensive(),
        _mixed(),
    )


#: Type alias: alias for the 4 default strategies tuple.
DEFAULT_FOUR_STRATEGIES: Final[tuple[StrategyProfile, ...]] = default_strategies()
"""Backwards-compatible alias. Computed lazily at module import time."""


def strategy_by_name(name: StrategyName) -> StrategyProfile:
    """Helper per recuperare la strategia di default per nome."""
    for s in default_strategies():
        if s.name == name:
            return s
    raise KeyError(f"Unknown default strategy: {name!r}")
