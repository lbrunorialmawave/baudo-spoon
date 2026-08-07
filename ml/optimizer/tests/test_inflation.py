"""Tests for the pure ``estimate_effective_cost`` function.

Covers:
* Monotonicity in ``num_participants``.
* Monotonicity in ``role_percentile``.
* Threshold effect: no inflation below the configured percentile.
* Cap: never exceed ``max_inflation_multiplier``.
* Edge case: extreme ``num_participants`` does not blow up the cost.
* Defensive: cost is never below the nominal listino.
"""

from __future__ import annotations

import math

import pytest

from ml.optimizer.inflation import (
    compute_role_percentile_map,
    estimate_effective_cost,
    inflation_multiplier,
    is_inflation_active,
)
from ml.optimizer.models import InflationConfig, Player

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> InflationConfig:
    return InflationConfig(
        inflation_percentile_threshold=0.7,
        max_inflation_multiplier=1.6,
        base_inflation_rate=0.05,
        baseline_participants=8,
    )


@pytest.fixture
def player_a() -> Player:
    return Player(
        player_id="a1",
        name="Alpha",
        role="A",
        real_team="Inter",
        cost=20,
        projected_score=8.0,
    )


# ---------------------------------------------------------------------------
# Threshold: no inflation below
# ---------------------------------------------------------------------------


def test_no_inflation_below_threshold(cfg: InflationConfig, player_a: Player) -> None:
    """Players at or below the percentile threshold pay listino, no matter the league size."""
    for n in (1, 8, 20, 50, 100):
        cost = estimate_effective_cost(
            player_a, role_percentile=0.7, num_participants=n, config=cfg
        )
        assert cost == pytest.approx(float(player_a.cost)), (
            f"expected listino at threshold for n={n}, got {cost}"
        )
        cost_below = estimate_effective_cost(
            player_a, role_percentile=0.5, num_participants=n, config=cfg
        )
        assert cost_below == pytest.approx(float(player_a.cost))


def test_threshold_exact_is_inactive(cfg: InflationConfig) -> None:
    assert is_inflation_active(0.7, cfg) is False
    assert is_inflation_active(0.7001, cfg) is True


# ---------------------------------------------------------------------------
# Monotonicity in num_participants
# ---------------------------------------------------------------------------


def test_monotonic_in_num_participants(cfg: InflationConfig, player_a: Player) -> None:
    perc = 0.9
    prev = estimate_effective_cost(player_a, perc, 1, cfg)
    for n in (8, 10, 15, 20, 50, 100):
        c = estimate_effective_cost(player_a, perc, n, cfg)
        assert c >= prev
        prev = c


def test_baseline_no_inflation(cfg: InflationConfig, player_a: Player) -> None:
    """At baseline_participants, the multiplier is exactly 1.0 even for top players."""
    # Actually at baseline the extra_participants=0 so multiplier=1.0, no inflation.
    # To be safe, we test that num_participants == baseline_participants -> cost == listino.
    c = estimate_effective_cost(player_a, 1.0, cfg.baseline_participants, cfg)
    assert c == pytest.approx(float(player_a.cost))


# ---------------------------------------------------------------------------
# Monotonicity in role_percentile
# ---------------------------------------------------------------------------


def test_monotonic_in_role_percentile(cfg: InflationConfig, player_a: Player) -> None:
    n = 20
    prev = estimate_effective_cost(player_a, 0.0, n, cfg)
    for p in [0.3, 0.5, 0.71, 0.8, 0.9, 1.0]:
        c = estimate_effective_cost(player_a, p, n, cfg)
        assert c >= prev
        prev = c


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------


def test_multiplier_respects_cap(cfg: InflationConfig, player_a: Player) -> None:
    """Even with huge num_participants, multiplier never exceeds cap."""
    m = inflation_multiplier(1.0, 10_000, cfg)
    assert m <= cfg.max_inflation_multiplier + 1e-9


def test_cap_at_extreme_participants(player_a: Player) -> None:
    cfg = InflationConfig(
        inflation_percentile_threshold=0.7,
        max_inflation_multiplier=1.6,
        base_inflation_rate=0.05,
        baseline_participants=8,
    )
    cost = estimate_effective_cost(player_a, 1.0, 10_000, cfg)
    assert cost == pytest.approx(player_a.cost * cfg.max_inflation_multiplier)


def test_cap_exact_formula_at_extremes() -> None:
    """At percentile=1.0 the multiplier is bounded by cap even with small extra participants."""
    cfg = InflationConfig(
        inflation_percentile_threshold=0.7,
        max_inflation_multiplier=1.6,
        base_inflation_rate=10.0,  # huge rate to force the cap
        baseline_participants=8,
    )
    cost = estimate_effective_cost(
        Player("x", "X", "D", "Roma", 10, 7.0),
        1.0,
        9,
        cfg,
    )
    # extra=1, headroom=1.0, raw=1+10*1*1=11 -> capped to 1.6
    assert cost == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# Cost floor: never below listino
# ---------------------------------------------------------------------------


def test_effective_cost_never_below_listino(
    cfg: InflationConfig, player_a: Player
) -> None:
    for n in (1, 8, 100):
        for p in (0.0, 0.5, 0.7, 0.9, 1.0, -0.1, 1.5):  # including out-of-range
            c = estimate_effective_cost(player_a, p, n, cfg)
            assert c >= player_a.cost


# ---------------------------------------------------------------------------
# Defensive / out-of-range
# ---------------------------------------------------------------------------


def test_percentile_clamped_to_unit_interval(cfg: InflationConfig) -> None:
    p = Player("p1", "P", "D", "Atalanta", 15, 6.5)
    c1 = estimate_effective_cost(p, 1.5, 20, cfg)  # > 1
    c2 = estimate_effective_cost(p, 1.0, 20, cfg)
    assert c1 == pytest.approx(c2)
    c3 = estimate_effective_cost(p, -0.5, 20, cfg)  # < 0
    c4 = estimate_effective_cost(p, 0.0, 20, cfg)
    assert c3 == pytest.approx(c4)


def test_zero_cost_player_returns_zero(cfg: InflationConfig) -> None:
    p = Player("z", "Z", "P", "Lazio", 0, 5.0)
    for n in (1, 8, 20):
        for perc in (0.0, 0.5, 1.0):
            assert estimate_effective_cost(p, perc, n, cfg) == 0


def test_num_participants_must_be_positive(
    cfg: InflationConfig, player_a: Player
) -> None:
    with pytest.raises(ValueError):
        estimate_effective_cost(player_a, 0.5, 0, cfg)


# ---------------------------------------------------------------------------
# role_percentile_map
# ---------------------------------------------------------------------------


def test_percentile_map_basic() -> None:
    ps = [
        Player("1", "A", "A", "T1", 10, 5.0),
        Player("2", "B", "A", "T2", 10, 7.0),
        Player("3", "C", "A", "T3", 10, 6.0),
    ]
    m = compute_role_percentile_map(ps)
    assert m["2"] == pytest.approx(1.0)  # best
    assert m["1"] == pytest.approx(0.0)  # worst
    assert m["3"] == pytest.approx(0.5)  # middle


def test_percentile_map_single_player() -> None:
    p = Player("solo", "Solo", "P", "Team", 5, 5.0)
    m = compute_role_percentile_map([p])
    assert m == {"solo": 1.0}


def test_percentile_map_per_role() -> None:
    ps = [
        Player("a1", "A1", "A", "T", 10, 9.0),
        Player("a2", "A2", "A", "T", 10, 7.0),
        Player("d1", "D1", "D", "T", 10, 6.0),
        Player("d2", "D2", "D", "T", 10, 8.0),
    ]
    m = compute_role_percentile_map(ps)
    assert m["a1"] == 1.0 and m["a2"] == 0.0
    assert m["d1"] == 0.0 and m["d2"] == 1.0


def test_percentile_map_empty() -> None:
    assert compute_role_percentile_map([]) == {}


def test_percentile_map_deterministic_on_ties() -> None:
    ps = [
        Player("a", "A", "A", "T", 10, 5.0),
        Player("b", "B", "A", "T", 10, 5.0),
    ]
    m = compute_role_percentile_map(ps)
    # Same score -> tie broken by player_id; 'a' < 'b' -> 'a' first -> rank 0
    assert m == {"a": 0.0, "b": 1.0}


# ---------------------------------------------------------------------------
# Pure-function contract: repeated calls yield same result
# ---------------------------------------------------------------------------


def test_pure_function_determinism(cfg: InflationConfig, player_a: Player) -> None:
    a = estimate_effective_cost(player_a, 0.9, 12, cfg)
    b = estimate_effective_cost(player_a, 0.9, 12, cfg)
    assert a == b
    assert not math.isnan(a)
