"""Auction VAR must reflect the already-shrunk projected_score."""

from __future__ import annotations

import pytest

from ml.auction.var import VAR, ReplacementLevel, VarEngine
from ml.sample_reliability.cohort import (
    COHORT_LIMITED,
    COHORT_STANDARD,
    RELIABILITY_WEIGHT_BY_COHORT,
)


def test_var_uses_projected_score_directly() -> None:
    repl = ReplacementLevel(role="C", score=6.0, n_players_used=10)
    raw_var = VAR.compute(
        player_id="raw", role="C", projected_score=8.5, replacement_level=repl
    )
    shrunk_var = VAR.compute(
        player_id="shrunk", role="C", projected_score=7.0, replacement_level=repl
    )
    assert raw_var.var_score == pytest.approx(2.5)
    assert shrunk_var.var_score == pytest.approx(1.0)
    assert shrunk_var.var_score < raw_var.var_score


def test_var_engine_assigns_lower_var_to_shrunk_limited() -> None:
    """STANDARD at 8.5 must have higher VAR than LIMITED at 7.0 (display value)."""
    players = [
        {
            "player_id": "std-1",
            "role": "C",
            "projected_score": 8.5,
            "cost": 20,
            "sample_cohort": COHORT_STANDARD,
            "reliability_weight": RELIABILITY_WEIGHT_BY_COHORT[COHORT_STANDARD],
        },
        {
            "player_id": "lim-1",
            "role": "C",
            "projected_score": 7.0,
            "cost": 20,
            "sample_cohort": COHORT_LIMITED,
            "reliability_weight": RELIABILITY_WEIGHT_BY_COHORT[COHORT_LIMITED],
        },
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 6.0, "cost": 5} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 6.0, "cost": 5} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 6.0, "cost": 5} for i in range(12)],
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 6.0, "cost": 5} for i in range(10)],
    ]

    engine = VarEngine(total_budget=500, num_participants=8)
    results = engine.evaluate(players)
    by_id = {r.player_id: r for r in results}

    assert by_id["std-1"].var_score > by_id["lim-1"].var_score
