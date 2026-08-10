"""Phase 4 tests: Monte Carlo simulations."""
import numpy as np
import pytest
from ml.simulations.monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
    N_SIMULATIONS,
    MIN_RESIDUALS,
)


@pytest.fixture
def fitted_simulator():
    residuals = []
    np.random.seed(0)
    for player_id in ["p1", "p2", "p3"]:
        for _ in range(20):
            residuals.append(
                {"player_id": player_id, "role": "A", "residual": float(np.random.normal(0, 0.5))}
            )
    for _ in range(30):
        residuals.append(
            {"player_id": "p_other", "role": "D", "residual": float(np.random.normal(0, 0.3))}
        )
    sim = MonteCarloSimulator(random_seed=42)
    sim.fit(residuals)
    return sim


class TestMonteCarloSimulator:
    def test_simulate_returns_correct_shape(self, fitted_simulator):
        result = fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert isinstance(result, SimulationResult)
        assert len(result.simulated_scores) == N_SIMULATIONS

    def test_scores_clipped_to_valid_range(self, fitted_simulator):
        result = fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert result.simulated_scores.min() >= 1.0
        assert result.simulated_scores.max() <= 10.0

    def test_percentiles_ordered(self, fitted_simulator):
        result = fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert (
            result.p10_score
            <= result.p25_score
            <= result.p50_score
            <= result.p75_score
            <= result.p90_score
        )

    def test_upside_and_downside_non_negative(self, fitted_simulator):
        result = fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert result.upside_potential >= 0
        assert result.downside_risk >= 0

    def test_bootstrap_player_method_used_when_sufficient(self, fitted_simulator):
        result = fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert result.sampling_method == "bootstrap_player"

    def test_fallback_to_role_when_player_residuals_insufficient(self, fitted_simulator):
        # "unknown_p" has no residuals; role "A" has sufficient
        result = fitted_simulator.simulate("unknown_p", predicted_score=6.5, role="A")
        assert result.sampling_method == "bootstrap_role"

    def test_parametric_fallback_when_no_data(self, fitted_simulator):
        result = fitted_simulator.simulate("unknown_p", predicted_score=6.5, role="GK")
        assert result.sampling_method == "parametric"

    def test_simulate_many_returns_one_per_player(self, fitted_simulator):
        players = [
            {"player_id": "p1", "predicted_score": 6.5, "role": "A"},
            {"player_id": "p2", "predicted_score": 5.8, "role": "A"},
        ]
        results = fitted_simulator.simulate_many(players, n_simulations=100)
        assert len(results) == 2

    def test_reproducible_with_same_seed(self):
        residuals = [
            {"player_id": "x", "role": "C", "residual": float(v)}
            for v in np.random.normal(0, 0.5, 20)
        ]
        s1 = MonteCarloSimulator(random_seed=42).fit(residuals)
        s2 = MonteCarloSimulator(random_seed=42).fit(residuals)
        r1 = s1.simulate("x", 6.0, "C")
        r2 = s2.simulate("x", 6.0, "C")
        np.testing.assert_array_equal(r1.simulated_scores, r2.simulated_scores)

    def test_simulate_before_fit_raises_or_uses_parametric(self):
        sim = MonteCarloSimulator(random_seed=42)
        # not fitted — should fall back to parametric (no crash)
        result = sim.simulate("p1", 6.0, "A")
        assert result.sampling_method == "parametric"


    def test_out_of_range_predicted_score_emits_warning(self, fitted_simulator, caplog):
        """predicted_score outside clip range must log a warning (observability)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ml.simulations.monte_carlo"):
            result = fitted_simulator.simulate("p1", predicted_score=17.2, role="A")
        assert result.simulated_scores.max() <= 10.0
        assert any(
            "outside clip range" in rec.message and "17.200" in rec.message
            for rec in caplog.records
        )

    def test_in_range_predicted_score_no_false_warning(self, fitted_simulator, caplog):
        """Normal predicted_score must not emit the out-of-range warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ml.simulations.monte_carlo"):
            fitted_simulator.simulate("p1", predicted_score=6.5, role="A")
        assert not any("outside clip range" in rec.message for rec in caplog.records)
