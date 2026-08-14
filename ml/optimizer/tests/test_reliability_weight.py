"""End-to-end: reliability_weight must influence the ILP objective.

Two midfielders with identical cost and projected_score but different
cohort weights — the solver must prefer the STANDARD player when the
rest of the squad is forced to fill with lower-scoring fillers.
"""

from __future__ import annotations

import pytest

from ml.optimizer.models import (
    Formation,
    InflationConfig,
    OptimizationConfig,
    Player,
    StrategyProfile,
)
from ml.optimizer.optimizer import optimize_squad
from ml.sample_reliability.cohort import (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
    RELIABILITY_WEIGHT_BY_COHORT,
)


def _cfg(**kwargs) -> OptimizationConfig:
    defaults = dict(
        budget=500,
        formations=[
            Formation("3-4-3", 3, 4, 3),
            Formation("4-3-3", 4, 3, 3),
            Formation("4-4-2", 4, 4, 2),
            Formation("3-5-2", 3, 5, 2),
        ],
        num_participants=8,
        max_players_per_team=8,
        big_teams=frozenset({"Inter", "Milan", "Juventus", "Napoli"}),
        big_teams_cap=25,
        min_distinct_teams=3,
        inflation_config=InflationConfig(),
    )
    defaults.update(kwargs)
    return OptimizationConfig(**defaults)


def _balanced() -> StrategyProfile:
    return StrategyProfile(
        name="BALANCED",
        role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
    )


def _fillers() -> list[Player]:
    """3P / 8D / 6C / 6A at score 5.5 — leaves 2 C slots for candidates."""
    pool: list[Player] = []
    teams = [f"T{i}" for i in range(12)]
    cid = 0
    for role, n in (("P", 3), ("D", 8), ("A", 6)):
        for i in range(n):
            cid += 1
            pool.append(
                Player(
                    player_id=f"{role}{cid}",
                    name=f"{role}{cid}",
                    role=role,  # type: ignore[arg-type]
                    real_team=teams[i % len(teams)],
                    cost=5,
                    projected_score=5.5,
                    reliability_weight=1.0,
                )
            )
    for i in range(7):
        cid += 1
        pool.append(
            Player(
                player_id=f"Cfill{cid}",
                name=f"Cfill{cid}",
                role="C",
                real_team=teams[(i + 3) % len(teams)],
                cost=5,
                projected_score=5.5,
                reliability_weight=1.0,
            )
        )
    return pool


def test_solver_prefers_standard_over_limited_at_equal_score() -> None:
    pool = _fillers()
    standard = Player(
        player_id="C_STD",
        name="Standard Mid",
        role="C",
        real_team="T0",
        cost=5,
        projected_score=8.0,
        reliability_weight=RELIABILITY_WEIGHT_BY_COHORT[COHORT_STANDARD],
    )
    limited = Player(
        player_id="C_LIM",
        name="Limited Mid",
        role="C",
        real_team="T1",
        cost=5,
        projected_score=8.0,
        reliability_weight=RELIABILITY_WEIGHT_BY_COHORT[COHORT_LIMITED],
    )
    pool.extend([standard, limited])

    res = optimize_squad(pool, _cfg(), _balanced())
    assert res.status == "OPTIMAL"
    selected_ids = {p.player_id for p in res.squad}
    assert "C_STD" in selected_ids, "STANDARD midfielder must be selected"
    assert "C_LIM" not in selected_ids, "LIMITED midfielder must lose to STANDARD at equal score"


def test_solver_prefers_standard_over_insufficient_at_equal_score() -> None:
    pool = _fillers()
    standard = Player(
        player_id="C_STD",
        name="Standard Mid",
        role="C",
        real_team="T0",
        cost=5,
        projected_score=8.0,
        reliability_weight=RELIABILITY_WEIGHT_BY_COHORT[COHORT_STANDARD],
    )
    insuf = Player(
        player_id="C_INS",
        name="Insufficient Mid",
        role="C",
        real_team="T1",
        cost=5,
        projected_score=8.0,
        reliability_weight=RELIABILITY_WEIGHT_BY_COHORT[COHORT_INSUFFICIENT],
    )
    pool.extend([standard, insuf])

    res = optimize_squad(pool, _cfg(), _balanced())
    assert res.status == "OPTIMAL"
    selected_ids = {p.player_id for p in res.squad}
    assert "C_STD" in selected_ids
    assert "C_INS" not in selected_ids


def test_none_reliability_weight_behaves_as_one() -> None:
    pool = _fillers()
    with_weight = Player(
        player_id="C_W",
        name="Weighted",
        role="C",
        real_team="T0",
        cost=5,
        projected_score=8.0,
        reliability_weight=1.0,
    )
    without = Player(
        player_id="C_NONE",
        name="Legacy",
        role="C",
        real_team="T1",
        cost=5,
        projected_score=8.0,
        reliability_weight=None,
    )
    pool.extend([with_weight, without])

    res = optimize_squad(pool, _cfg(), _balanced())
    assert res.status == "OPTIMAL"
    assert res.role_breakdown["C"] == 8
