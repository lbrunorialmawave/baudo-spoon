"""Auction VAR: display-shrunk projected_score + decision reliability weight (ADR 0001)."""

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


def test_var_engine_default_applies_reliability_weight() -> None:
    """ADR 0001: default apply_reliability_weight=True multiplies the score."""
    players = [
        {
            "player_id": "full",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 1.0,
        },
        {
            "player_id": "half",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 0.5,
        },
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 5.0} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 5.0} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 5.0} for i in range(12)],
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 5.0} for i in range(10)],
    ]
    engine = VarEngine(total_budget=500, num_participants=8)  # defaults only
    results = engine.evaluate(players)
    by_id = {r.player_id: r for r in results}
    assert by_id["full"].var_score > by_id["half"].var_score


def test_var_engine_can_disable_reliability_weight() -> None:
    """Explicit False restores pre-hardening ranking (same score at same projected)."""
    players = [
        {
            "player_id": "a",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 0.5,
        },
        {
            "player_id": "b",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 1.0,
        },
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 5.0} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 5.0} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 5.0} for i in range(12)],
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 5.0} for i in range(10)],
    ]
    engine = VarEngine(
        total_budget=500, num_participants=8, apply_reliability_weight=False
    )
    results = engine.evaluate(players)
    by_id = {r.player_id: r for r in results}
    assert by_id["a"].var_score == pytest.approx(by_id["b"].var_score)


def test_adzic_style_not_in_top_decile_with_defaults() -> None:
    """End-to-end guard: Adzic-style LIMITED with high display score is ranked down.

    VarEngine() with no overrides must apply reliability_weight so a LIMITED
    player at 0.65 weight does not outrank a STANDARD peer at similar raw score
    enough to monopolise the top of the list.
    """
    players = [
        {
            "player_id": "adzic-163",
            "role": "A",
            "projected_score": 7.8,  # still high after display shrink
            "reliability_weight": 0.40,  # continuous ~163'
            "sample_cohort": COHORT_LIMITED,
        },
        {
            "player_id": "std-fwd",
            "role": "A",
            "projected_score": 7.1,
            "reliability_weight": 1.0,
            "sample_cohort": COHORT_STANDARD,
        },
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 6.2, "reliability_weight": 1.0} for i in range(10)],
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 6.0, "reliability_weight": 1.0} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 6.0, "reliability_weight": 1.0} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 6.0, "reliability_weight": 1.0} for i in range(12)],
    ]
    engine = VarEngine(total_budget=500, num_participants=8)
    results = engine.evaluate(players)
    # Rank by var_score within role A
    role_a = [r for r in results if r.role == "A"]
    role_a_sorted = sorted(role_a, key=lambda r: r.var_score, reverse=True)
    top_decile_n = max(1, len(role_a_sorted) // 10)
    top_ids = {r.player_id for r in role_a_sorted[:top_decile_n]}
    # With weight 0.40, effective score ≈ 3.12 → well below STANDARD 7.1
    assert "adzic-163" not in top_ids
    assert role_a_sorted[0].player_id == "std-fwd"


def test_var_engine_applies_reliability_weight_when_enabled() -> None:
    """WS3 Option B: reliability_weight multiplies the decision score."""
    players = [
        {
            "player_id": "full",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 1.0,
        },
        {
            "player_id": "half",
            "role": "C",
            "projected_score": 8.0,
            "reliability_weight": 0.5,
        },
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 5.0} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 5.0} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 5.0} for i in range(12)],
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 5.0} for i in range(10)],
    ]
    engine = VarEngine(
        total_budget=500,
        num_participants=8,
        apply_reliability_weight=True,
    )
    results = engine.evaluate(players)
    by_id = {r.player_id: r for r in results}
    assert by_id["full"].var_score > by_id["half"].var_score


def test_var_engine_applies_risk_aversion() -> None:
    players = [
        {
            "player_id": "sure",
            "role": "C",
            "projected_score": 8.0,
            "prediction_std": 0.1,
        },
        {
            "player_id": "noisy",
            "role": "C",
            "projected_score": 8.0,
            "prediction_std": 2.0,
        },
        *[{"player_id": f"c{i}", "role": "C", "projected_score": 5.0} for i in range(15)],
        *[{"player_id": f"p{i}", "role": "P", "projected_score": 5.0} for i in range(5)],
        *[{"player_id": f"d{i}", "role": "D", "projected_score": 5.0} for i in range(12)],
        *[{"player_id": f"a{i}", "role": "A", "projected_score": 5.0} for i in range(10)],
    ]
    engine = VarEngine(total_budget=500, num_participants=8, risk_aversion=0.5)
    results = engine.evaluate(players)
    by_id = {r.player_id: r for r in results}
    assert by_id["sure"].var_score > by_id["noisy"].var_score
