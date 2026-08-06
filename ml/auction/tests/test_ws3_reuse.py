"""WS3 — cross-module reuse (Optimizer → Auction).

Covers plan §7 items 1–5 at the domain layer.
"""

from __future__ import annotations

import pytest

from ml.auction.alternatives import (
    max_affordable_bid,
    pareto_diversify,
    strategy_price_cap,
    suggest_alternatives,
)
from ml.auction.completion_probability import (
    estimate_all_completion_probabilities,
    estimate_participant_completion_probability,
)
from ml.auction.models import AlternativesConfig, AuctionConfig, ParticipantSetup, ValuationMode
from ml.auction.orchestrator import get_auction_summary, initialize_auction, record_assignment
from ml.auction.var import VarEngine
from ml.optimizer.models import Player


def _ps(n: int = 2, budget: int = 500) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(participant_id=f"u{i}", display_name=f"U{i}", budget_initial=budget)
        for i in range(1, n + 1)
    ]


def _p(pid: str, role: str, cost: int = 10, score: float = 6.0, **kw: object) -> Player:
    return Player(
        player_id=pid,
        name=pid,
        real_team="T",
        role=role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# #1 completion probability
# ---------------------------------------------------------------------------


def test_completion_probability_full_roster_is_one() -> None:
    cfg = AuctionConfig(num_participants=2, role_quotas={"P": 1, "D": 1, "C": 1, "A": 1})
    pool = [_p("p1", "P"), _p("d1", "D"), _p("c1", "C"), _p("a1", "A")]
    state = initialize_auction(_ps(2, budget=100), cfg, pool)
    # Fill all slots for u1
    for pid, role in [("p1", "P"), ("d1", "D"), ("c1", "C"), ("a1", "A")]:
        # prices must respect credit reserve
        r = record_assignment(state, pid, "u1", 1)
        assert r.success, r.rejection_reason
    prob = estimate_participant_completion_probability(state, "u1")
    assert prob == 1.0


def test_completion_probability_in_summary() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_p(f"{r}{i}", r, cost=5, score=5.0) for r in "PDCA" for i in range(3)]
    state = initialize_auction(_ps(2, budget=300), cfg, pool)
    summary = get_auction_summary(state)
    assert summary.completion_probability is not None
    assert "u1" in summary.completion_probability
    assert 0.0 <= summary.completion_probability["u1"] <= 1.0


def test_all_completion_probabilities_keys() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_p("p1", "P"), _p("d1", "D")]
    state = initialize_auction(_ps(2), cfg, pool)
    probs = estimate_all_completion_probabilities(state)
    assert set(probs) == {"u1", "u2"}


# ---------------------------------------------------------------------------
# #2 hybrid_blend in VarEngine
# ---------------------------------------------------------------------------


def test_hybrid_blend_zero_is_pure_projected() -> None:
    engine = VarEngine(hybrid_blend=0.0, total_budget=500)
    players = [
        {"player_id": "a", "role": "A", "projected_score": 8.0, "fp_ibrido": 4.0},
        {"player_id": "b", "role": "A", "projected_score": 6.0, "fp_ibrido": 9.0},
        {"player_id": "c", "role": "A", "projected_score": 5.0},
    ]
    results = {r.player_id: r for r in engine.evaluate(players)}
    # With blend=0, ranking by projected_score: a > b > c
    assert results["a"].var_score > results["b"].var_score
    assert results["b"].var_score > results["c"].var_score


def test_hybrid_blend_pulls_toward_fp_ibrido() -> None:
    """High fp_ibrido with blend=1.0 should dominate pure projected ranking."""
    base = [
        {"player_id": "low_fp", "role": "A", "projected_score": 9.0, "fp_ibrido": 4.0},
        {"player_id": "high_fp", "role": "A", "projected_score": 5.0, "fp_ibrido": 9.5},
        {"player_id": "mid", "role": "A", "projected_score": 6.0, "fp_ibrido": 6.0},
    ]
    pure = {r.player_id: r for r in VarEngine(hybrid_blend=0.0).evaluate(base)}
    blended = {r.player_id: r for r in VarEngine(hybrid_blend=1.0).evaluate(base)}
    # pure: low_fp has higher projected → higher VAR
    assert pure["low_fp"].var_score > pure["high_fp"].var_score
    # blend=1: high_fp uses only fp_ibrido → higher VAR
    assert blended["high_fp"].var_score > blended["low_fp"].var_score


def test_hybrid_blend_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="hybrid_blend"):
        VarEngine(hybrid_blend=1.5)


# ---------------------------------------------------------------------------
# #3 Pareto diversify
# ---------------------------------------------------------------------------


def test_pareto_diversify_returns_non_dominated() -> None:
    candidates = [
        _p("cheap_good", "D", cost=5, score=7.0),
        _p("expensive_great", "D", cost=30, score=9.0),
        _p("dominated", "D", cost=25, score=6.0),  # worse score & higher price than cheap_good
        _p("mid", "D", cost=12, score=7.5),
    ]
    prices = {p.player_id: float(p.cost) for p in candidates}
    front = pareto_diversify(candidates, prices, max_points=5)
    ids = {p.player_id for p in front}
    assert "dominated" not in ids
    assert "cheap_good" in ids or "mid" in ids


def test_suggest_alternatives_includes_diversified() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [
        _p("t", "D", cost=20, score=7.0),
        _p("a", "D", cost=8, score=6.5),
        _p("b", "D", cost=15, score=7.2),
        _p("c", "D", cost=12, score=6.8),
        _p("x", "C", cost=10, score=6.0),
    ]
    state = initialize_auction(_ps(2), cfg, pool)
    suggestion = suggest_alternatives(
        target=pool[0],
        available_pool=state.available_pool,
        state=state,
        config=AlternativesConfig(),
        diversify=True,
    )
    assert suggestion.low_cost_alternative is not None or suggestion.closest_alternative is not None
    # diversified is a tuple (may be empty if pool tiny after exclude)
    assert isinstance(suggestion.diversified_alternatives, tuple)


# ---------------------------------------------------------------------------
# #4 max affordable bid
# ---------------------------------------------------------------------------


def test_max_affordable_bid_credit_reserve() -> None:
    # total slots 25, residual budget 100, 0 filled → max = 100 - 24 = 76
    cfg = AuctionConfig(num_participants=2, budget_initial=100)
    pool = [_p("p1", "P")]
    state = initialize_auction(
        [ParticipantSetup("u1", "U1", 100), ParticipantSetup("u2", "U2", 100)],
        cfg,
        pool,
    )
    assert max_affordable_bid(state, "u1") == 100 - 24


def test_suggest_alternatives_exposes_max_bid() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_p("t", "D"), _p("a", "D")]
    state = initialize_auction(_ps(2, budget=200), cfg, pool)
    suggestion = suggest_alternatives(
        target=pool[0],
        available_pool=state.available_pool,
        state=state,
        config=AlternativesConfig(),
        participant_id="u1",
    )
    assert suggestion.max_affordable_bid is not None
    assert suggestion.max_affordable_bid == 200 - 24


# ---------------------------------------------------------------------------
# #5 strategy price cap
# ---------------------------------------------------------------------------


def test_strategy_price_cap_defensive_boosts_d() -> None:
    p = _p("d1", "D", cost=20)
    cap_bal = strategy_price_cap(p, 20.0, "BALANCED")
    cap_def = strategy_price_cap(p, 20.0, "SUPER_DEFENSIVE")
    assert cap_bal == 20
    assert cap_def is not None and cap_def > cap_bal


def test_strategy_price_cap_none_without_strategy() -> None:
    assert strategy_price_cap(_p("a", "A"), 15.0, None) is None


def test_suggest_with_strategy_name() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_p("t", "A", cost=25, score=8.0), _p("a", "A", cost=10, score=6.0)]
    state = initialize_auction(_ps(2), cfg, pool)
    suggestion = suggest_alternatives(
        target=pool[0],
        available_pool=state.available_pool,
        state=state,
        config=AlternativesConfig(),
        strategy_name="SUPER_OFFENSIVE",
    )
    assert suggestion.strategy_price_cap is not None
    assert suggestion.strategy_price_cap >= 1
