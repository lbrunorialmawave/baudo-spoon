"""Tests for the 4 default strategies and StrategyProfile behavior."""

from __future__ import annotations

import pytest

from ml.optimizer.models import OptimizationConfig, StrategyProfile
from ml.optimizer.strategies import (
    DEFAULT_FOUR_STRATEGIES,
    DEFAULT_MAX_TOP_TIER_PLAYERS,
    DEFAULT_SUPER_DEFENSIVE_SHARE,
    DEFAULT_SUPER_OFFENSIVE_SHARE,
    DEFAULT_TOP_TIER_COST_THRESHOLD,
    default_strategies,
    strategy_by_name,
)


def test_default_strategies_returns_4() -> None:
    strategies = default_strategies()
    assert len(strategies) == 4
    names = {s.name for s in strategies}
    assert names == {"BALANCED", "SUPER_DEFENSIVE", "SUPER_OFFENSIVE", "MIXED"}


def test_default_four_strategies_alias() -> None:
    assert DEFAULT_FOUR_STRATEGIES == default_strategies()


def test_balanced_has_unit_weights() -> None:
    bal = strategy_by_name("BALANCED")
    for w in bal.role_weight.values():
        assert w == 1.0
    assert bal.min_budget_share_by_roles is None
    assert bal.max_top_tier_players is None


def test_super_defensive_prefers_defense() -> None:
    sd = strategy_by_name("SUPER_DEFENSIVE")
    assert sd.role_weight["D"] > sd.role_weight["A"]
    assert sd.min_budget_share_by_roles is not None
    roles, share = sd.min_budget_share_by_roles
    assert roles == frozenset({"P", "D"})
    assert share == DEFAULT_SUPER_DEFENSIVE_SHARE


def test_super_offensive_prefers_attack() -> None:
    so = strategy_by_name("SUPER_OFFENSIVE")
    assert so.role_weight["A"] > so.role_weight["D"]
    assert so.min_budget_share_by_roles is not None
    roles, share = so.min_budget_share_by_roles
    assert roles == frozenset({"C", "A"})
    assert share == DEFAULT_SUPER_OFFENSIVE_SHARE


def test_mixed_caps_top_tier() -> None:
    mixed = strategy_by_name("MIXED")
    assert mixed.max_top_tier_players == DEFAULT_MAX_TOP_TIER_PLAYERS
    assert mixed.top_tier_cost_threshold == float(DEFAULT_TOP_TIER_COST_THRESHOLD)


def test_strategy_by_name_unknown_raises() -> None:
    with pytest.raises(KeyError):
        strategy_by_name("UNKNOWN")  # type: ignore[arg-type]


def test_optimization_config_default_strategies() -> None:
    from ml.optimizer.models import Formation

    cfg = OptimizationConfig(
        budget=500, formations=[Formation("3-4-3", 3, 4, 3)], num_participants=8
    )
    assert len(cfg.strategies) == 4


def test_strategy_profile_with_optional_fields() -> None:
    sp = StrategyProfile(
        name="BALANCED",
        role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
        min_budget_share_by_roles=(frozenset({"D"}), 0.30),
    )
    assert sp.min_budget_share_by_roles is not None
    roles, share = sp.min_budget_share_by_roles
    assert roles == frozenset({"D"})
    assert share == 0.30
