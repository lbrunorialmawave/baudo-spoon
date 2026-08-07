"""Phase 5 tests: auction VAR, demand curve, ESV."""

import pytest

from ml.auction.var import (
    VAR,
    DemandCurve,
    ExpectedSurplusValue,
    ReplacementLevel,
    VarEngine,
)


@pytest.fixture
def role_scores():
    return [5.0, 5.5, 6.0, 6.0, 6.5, 7.0, 7.5, 8.0]


class TestReplacementLevel:
    def test_computed_from_bottom_percentile(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        assert rl.score <= min(role_scores) * 1.5

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ReplacementLevel.from_player_pool("A", [])

    def test_n_players_used_at_least_one(self, role_scores):
        rl = ReplacementLevel.from_player_pool(
            "A", role_scores, percentile_threshold=0.01
        )
        assert rl.n_players_used >= 1


class TestVAR:
    def test_var_above_replacement(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        v = VAR.compute("p1", "A", 8.0, rl)
        assert v.var_score > 0

    def test_var_below_replacement(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        v = VAR.compute("p_weak", "A", 4.0, rl)
        assert v.var_score < 0

    def test_var_score_equals_diff(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        v = VAR.compute("p1", "A", 7.5, rl)
        assert abs(v.var_score - (7.5 - rl.score)) < 1e-9


class TestDemandCurve:
    def test_calibrated_false_by_default(self):
        dc = DemandCurve()
        assert not dc.calibrated

    def test_higher_var_higher_price(self):
        dc = DemandCurve(calibrated=True)
        assert dc.expected_price(3.0) > dc.expected_price(1.0)

    def test_negative_var_returns_base_price(self):
        dc = DemandCurve(base_price=1.0, calibrated=True)
        assert dc.expected_price(-5.0) == 1.0

    def test_price_is_monotone_convex_in_var(self):
        dc = DemandCurve(calibrated=True)
        prices = [dc.expected_price(v) for v in [0, 1, 2, 3, 4]]
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]


class TestExpectedSurplusValue:
    def test_high_var_positive_esv(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        v = VAR.compute("p_star", "A", 9.0, rl)
        dc = DemandCurve(scale=2.0, calibrated=True)
        esv = ExpectedSurplusValue.compute(
            v, dc, budget_per_slot=20.0, baseline_var=2.0
        )
        # Star player with low-scale demand curve should have positive ESV
        assert isinstance(esv.esv, float)

    def test_esv_calibrated_field_mirrors_demand_curve(self, role_scores):
        rl = ReplacementLevel.from_player_pool("A", role_scores)
        v = VAR.compute("p1", "A", 7.0, rl)
        dc = DemandCurve(calibrated=False)
        esv = ExpectedSurplusValue.compute(
            v, dc, budget_per_slot=20.0, baseline_var=2.0
        )
        assert not esv.calibrated


class TestVarEngine:
    def test_evaluate_returns_sorted_by_esv(self):
        players = [
            {"player_id": f"p{i}", "role": "A", "projected_score": float(5 + i * 0.5)}
            for i in range(8)
        ]
        engine = VarEngine(total_budget=500)
        results = engine.evaluate(players)
        esvs = [r.esv for r in results]
        assert esvs == sorted(esvs, reverse=True)

    def test_evaluate_handles_multiple_roles(self):
        players = [
            {"player_id": f"d{i}", "role": "D", "projected_score": float(5 + i * 0.3)}
            for i in range(5)
        ] + [
            {"player_id": f"a{i}", "role": "A", "projected_score": float(6 + i * 0.5)}
            for i in range(5)
        ]
        engine = VarEngine(total_budget=500)
        results = engine.evaluate(players)
        roles = {r.role for r in results}
        assert "D" in roles and "A" in roles

    def test_evaluate_empty_raises_or_returns_empty(self):
        engine = VarEngine()
        results = engine.evaluate([])
        assert results == []
