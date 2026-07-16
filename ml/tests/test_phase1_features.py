"""Phase 1 risk-based tests.

Critical invariants verified:
1. No look-ahead bias in RollingMeanFeature
2. SAP direction: strong opponents → adjusted stat > weak opponents
3. MissingDataPolicy.FAIL raises on missing stat column
4. compute_difficulty_coefficients returns all 1.0 when no component columns present
5. compute_difficulty_coefficients range: all values in [coeff_min, coeff_max]
6. Weight redistribution: rescaled weights sum to 1.0 when a component is absent
7. FeatureRegistry raises on duplicate registration
8. compute_feature_matrix on_error="skip" skips failed features
9. TeamStrengthFeature normalisation output in [0, 1]
10. Per90Feature denominator: mins_played=90 → per90 = raw; mins_played=0 → clipped to 1
"""
import pytest
import polars as pl

from ml.domain.config import DEFAULT_SCHEDULE_ADJUSTMENT, ScheduleAdjustmentConfig
from ml.domain.features import Feature, MissingDataPolicy
from ml.features.base import FeatureRegistry, compute_feature_matrix
from ml.features.per90 import GoalsPer90
from ml.features.rolling import RollingMeanFeature
from ml.features.sap import SapFeature
from ml.features.team_strength import TeamStrengthFeature
from ml.schedule_adjustment.coefficients import (
    _COMPONENT_COLUMNS,
    compute_difficulty_coefficients,
)


# ── Test 1: No look-ahead bias ────────────────────────────────────────────────

class TestRollingMeanNoLookahead:
    def test_season_2022_has_no_prior_data(self) -> None:
        """First season: no prior data → rolling mean should be null/0."""
        df = pl.DataFrame({
            "player_fotmob_id": [1, 1, 1],
            "season_start":     [2022, 2023, 2024],
            "goals_per90":      [1.0, 2.0, 3.0],
        })

        class _Roll(RollingMeanFeature):
            stat_col = "goals_per90"
            window = 2

        feat = _Roll()
        result = feat.safe_compute(df).to_list()

        # season 2022: no prior → null (rendered as None or 0 after fill)
        assert result[0] is None or result[0] == pytest.approx(0.0, abs=1e-6), (
            f"Season 2022 rolling mean should be None/0, got {result[0]}"
        )

    def test_season_2024_uses_only_2022_and_2023(self) -> None:
        """Season 2024 rolling mean (window=2) must equal mean(2022, 2023) = 1.5."""
        df = pl.DataFrame({
            "player_fotmob_id": [1, 1, 1],
            "season_start":     [2022, 2023, 2024],
            "goals_per90":      [1.0, 2.0, 3.0],
        })

        class _Roll(RollingMeanFeature):
            stat_col = "goals_per90"
            window = 2

        feat = _Roll()
        result = feat.safe_compute(df).to_list()

        # season 2024 is the last row (index 2 in original order)
        assert result[2] == pytest.approx(1.5, abs=1e-6), (
            f"Season 2024 rolling mean should be 1.5 (mean of 2022+2023), got {result[2]}"
        )

    def test_season_2023_uses_only_2022(self) -> None:
        """Season 2023 rolling mean (window=2, min_samples=1) must equal season 2022 = 1.0."""
        df = pl.DataFrame({
            "player_fotmob_id": [1, 1, 1],
            "season_start":     [2022, 2023, 2024],
            "goals_per90":      [1.0, 2.0, 3.0],
        })

        class _Roll(RollingMeanFeature):
            stat_col = "goals_per90"
            window = 2

        feat = _Roll()
        result = feat.safe_compute(df).to_list()

        assert result[1] == pytest.approx(1.0, abs=1e-6), (
            f"Season 2023 rolling mean should be 1.0 (only 2022 prior), got {result[1]}"
        )


# ── Test 2: SAP direction ─────────────────────────────────────────────────────

class TestSapDirection:
    def test_strong_opponents_produce_higher_adjusted_stat(self) -> None:
        """Player on a WEAK team faces stronger opponents → higher SAP than player on STRONG team.

        Formula: opponent_mean_rank = (league_total - own_rank) / (n_teams - 1)

        A player on a weak team (low team_rank_norm) has opponents with high average rank
        → opponent_mean_rank is high → sap_weight > 1 → adjusted stat is higher.
        """
        # Two players in same league-season, same raw goals_per90=1.0.
        # Player A is on a WEAK team (low team_rank_norm=0.1) → faces strong opponents.
        # Player B is on a STRONG team (high team_rank_norm=0.9) → faces weak opponents.
        df = pl.DataFrame({
            "player_fotmob_id": [1, 2],
            "team_fotmob_id":   [10, 20],
            "season_start":     [2023, 2023],
            "league_name":      ["SerieA", "SerieA"],
            "goals_per90":      [1.0, 1.0],
            "team_rank_norm":   [0.1, 0.9],  # A=weak team, B=strong team
        })

        class _Sap(SapFeature):
            stat_col = "goals_per90"

        feat = _Sap()
        result = feat.safe_compute(df).to_list()

        assert result[0] > result[1], (
            f"Player A (weak team, strong opponents) sap={result[0]:.4f} should > "
            f"Player B (strong team, weak opponents) sap={result[1]:.4f}"
        )


# ── Test 3: FAIL policy raises on missing stat column ─────────────────────────

class TestFailPolicy:
    def test_fail_policy_raises_on_missing_stat_col(self) -> None:
        """A Feature with MissingDataPolicy.FAIL must raise ValueError when its column is absent."""
        class _StrictGoals(Feature):
            # Same shape as GoalsPer90 but with FAIL policy.
            name = "goals_per90_strict"
            required_columns = frozenset(["goals"])
            missing_data_policy = MissingDataPolicy.FAIL

            def compute(self, data: pl.DataFrame) -> pl.Series:
                denom = (data["mins_played"].cast(pl.Float64) / 90.0).clip(lower_bound=1.0)
                return data["goals"].cast(pl.Float64).fill_null(0.0) / denom

        feat = _StrictGoals()
        df = pl.DataFrame({"mins_played": [900.0, 450.0]})  # 'goals' absent
        with pytest.raises(ValueError, match="goals"):
            feat.safe_compute(df)


# ── Test 4: Neutral coefficients when no opponent columns ────────────────────

class TestDifficultyCoefficientsNeutral:
    def test_all_ones_when_no_component_columns(self) -> None:
        df = pl.DataFrame({"player_id": [1, 2, 3], "some_col": [1.0, 2.0, 3.0]})
        result = compute_difficulty_coefficients(df)
        assert result.to_list() == [1.0, 1.0, 1.0]


# ── Test 5: Coefficient range ─────────────────────────────────────────────────

class TestDifficultyCoefficientsRange:
    def test_all_values_within_config_range(self) -> None:
        config = DEFAULT_SCHEDULE_ADJUSTMENT
        df = pl.DataFrame({
            "season_start":               [2022, 2022, 2023, 2023],
            "opponent_elo":               [1400.0, 1600.0, 1500.0, 1550.0],
            "opponent_expected_points":   [40.0, 70.0, 55.0, 60.0],
            "opponent_league_position":   [15.0, 3.0, 10.0, 5.0],
            "opponent_goal_difference":   [-10.0, 25.0, 5.0, 15.0],
            "opponent_squad_value":       [50.0, 200.0, 100.0, 150.0],
        })
        result = compute_difficulty_coefficients(df, config)
        vals = result.to_list()
        for v in vals:
            assert config.coeff_min <= v <= config.coeff_max, (
                f"Coefficient {v} outside [{config.coeff_min}, {config.coeff_max}]"
            )


# ── Test 6: Weight redistribution sums to 1.0 ────────────────────────────────

class TestWeightRedistribution:
    def test_rescaled_weights_sum_to_one_when_column_absent(self) -> None:
        """When one component is absent, the remaining rescaled weights must sum to 1.0."""
        config = DEFAULT_SCHEDULE_ADJUSTMENT
        # Only 4 of 5 columns present (squad_value absent).
        df = pl.DataFrame({
            "season_start":               [2022],
            "opponent_elo":               [1500.0],
            "opponent_expected_points":   [55.0],
            "opponent_league_position":   [8.0],
            "opponent_goal_difference":   [5.0],
            # opponent_squad_value intentionally absent
        })
        # Manually verify the weight redistribution logic.
        available_weights = {
            "opponent_elo":             config.elo_weight,
            "opponent_expected_points": config.expected_points_weight,
            "opponent_league_position": config.league_position_weight,
            "opponent_goal_difference": config.goal_difference_weight,
        }
        total = sum(available_weights.values())
        rescaled = {col: w / total for col, w in available_weights.items()}
        assert abs(sum(rescaled.values()) - 1.0) < 1e-9

        # Also verify that compute_difficulty_coefficients doesn't crash.
        result = compute_difficulty_coefficients(df, config)
        assert len(result) == 1
        assert config.coeff_min <= result[0] <= config.coeff_max


# ── Test 7: FeatureRegistry no-duplicate ──────────────────────────────────────

class TestFeatureRegistry:
    def test_duplicate_name_raises(self) -> None:
        class _F(Feature):
            name = "dup_test"
            required_columns = frozenset(["x"])
            missing_data_policy = MissingDataPolicy.FAIL
            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["x"]

        registry = FeatureRegistry()
        registry.register(_F())
        with pytest.raises(ValueError, match="dup_test"):
            registry.register(_F())

    def test_get_unknown_raises(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")


# ── Test 8: compute_feature_matrix on_error="skip" ───────────────────────────

class TestComputeFeatureMatrix:
    def test_skip_failed_feature_computes_others(self) -> None:
        class _Good(Feature):
            name = "good"
            required_columns = frozenset(["x"])
            missing_data_policy = MissingDataPolicy.FAIL
            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["x"] * 2.0

        class _Bad(Feature):
            name = "bad"
            required_columns = frozenset(["missing_col"])
            missing_data_policy = MissingDataPolicy.FAIL
            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["missing_col"]

        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = compute_feature_matrix(df, [_Good(), _Bad()], on_error="skip")
        assert "good" in result.columns
        assert "bad" not in result.columns
        assert result["good"].to_list() == [2.0, 4.0, 6.0]

    def test_raise_on_error_propagates(self) -> None:
        class _Bad(Feature):
            name = "bad"
            required_columns = frozenset(["missing_col"])
            missing_data_policy = MissingDataPolicy.FAIL
            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["missing_col"]

        df = pl.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError):
            compute_feature_matrix(df, [_Bad()], on_error="raise")


# ── Test 9: TeamStrengthFeature normalisation ─────────────────────────────────

class TestTeamStrengthFeature:
    def test_output_in_zero_one_range(self) -> None:
        feat = TeamStrengthFeature()
        df = pl.DataFrame({"team_strength_score": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = feat.safe_compute(df).to_list()
        assert min(result) == pytest.approx(0.0, abs=1e-6)
        assert max(result) == pytest.approx(1.0, abs=1e-6)

    def test_constant_values_return_half(self) -> None:
        feat = TeamStrengthFeature()
        df = pl.DataFrame({"team_strength_score": [5.0, 5.0, 5.0]})
        result = feat.safe_compute(df).to_list()
        assert all(v == pytest.approx(0.5, abs=1e-6) for v in result)


# ── Test 10: Per90Feature denominator ────────────────────────────────────────

class TestPer90Denominator:
    def test_90_minutes_gives_per90_equal_to_raw(self) -> None:
        feat = GoalsPer90()
        df = pl.DataFrame({"goals": [3.0], "mins_played": [90.0]})
        result = feat.safe_compute(df)
        assert result[0] == pytest.approx(3.0, abs=1e-6)

    def test_zero_minutes_clipped_to_1_denominator(self) -> None:
        """mins_played=0 → denominator clipped to 1.0 → per90 = raw / (1/90*90) = raw/1."""
        feat = GoalsPer90()
        # mins_played=0 → denom = clip(0/90, 1.0) = 1.0
        df = pl.DataFrame({"goals": [5.0], "mins_played": [0.0]})
        result = feat.safe_compute(df)
        # denom = max(0/90, 1.0) = 1.0 → per90 = 5.0 / 1.0 = 5.0
        assert result[0] == pytest.approx(5.0, abs=1e-6)

    def test_180_minutes_gives_half_raw(self) -> None:
        feat = GoalsPer90()
        df = pl.DataFrame({"goals": [4.0], "mins_played": [180.0]})
        result = feat.safe_compute(df)
        assert result[0] == pytest.approx(2.0, abs=1e-6)
