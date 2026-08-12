"""Unit tests for :mod:`ml.preprocessing.role_features` (PR4).

These tests cover:
* Correctness of rolling means / ratios.
* Temporal isolation: the current season is never in the window
  (``closed='left'`` semantics).
* Behaviour with missing columns (silent skip, no crash).
* Linear-trend slope sign and stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.role_features import (
    RoleOpportunityFeatureTransformer,
    _rolling_slope,
    add_role_opportunity_features,
)


def _toy_frame() -> pd.DataFrame:
    """4 seasons, 2 players — a deterministic test fixture."""
    return pd.DataFrame(
        {
            "player_fotmob_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "season_start": [
                2020, 2021, 2022, 2023,
                2020, 2021, 2022, 2023,
            ],
            "mins_played": [
                500, 1000, 1500, 2000,   # player 1: increasing
                2000, 2000, 2000, 2000,  # player 2: flat
            ],
            "starts": [
                5, 15, 25, 30,
                30, 30, 30, 30,
            ],
            "appearances": [
                10, 20, 30, 35,
                35, 35, 35, 35,
            ],
        }
    )


class TestRoleOpportunityTransformer:
    def test_minutes_opp_is_strictly_historical(self) -> None:
        df = add_role_opportunity_features(_toy_frame(), opportunity_window=3)
        # Player 1 in 2020: no history → opp = NaN.
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2020)].iloc[0]
        assert np.isnan(row["mins_played_opp"])
        # Player 1 in 2023: history of [500, 1000, 1500] → mean = 1000.
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2023)].iloc[0]
        assert row["mins_played_opp"] == pytest.approx(1000.0)

    def test_current_season_excluded_from_window(self) -> None:
        """The 2023 opp value for player 1 must use 2020, 2021, 2022
        (mean=1000) and must NOT include the 2023 value of 2000."""
        df = add_role_opportunity_features(_toy_frame(), opportunity_window=3)
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2023)].iloc[0]
        assert row["mins_played_opp"] == pytest.approx(1000.0)
        # If the window were inclusive, the mean would be (500+1000+1500+2000)/4=1250.
        assert row["mins_played_opp"] != pytest.approx(1250.0)

    def test_minutes_trend_positive_for_increasing_player(self) -> None:
        df = add_role_opportunity_features(_toy_frame(), opportunity_window=4)
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2023)].iloc[0]
        # Strictly increasing minutes → positive slope.
        assert row["minutes_trend_opp"] > 0

    def test_minutes_trend_zero_for_flat_player(self) -> None:
        df = add_role_opportunity_features(_toy_frame(), opportunity_window=4)
        row = df[(df["player_fotmob_id"] == 2) & (df["season_start"] == 2023)].iloc[0]
        assert row["minutes_trend_opp"] == pytest.approx(0.0, abs=1e-9)

    def test_starts_per_appearance_smoothed(self) -> None:
        df = add_role_opportunity_features(_toy_frame(), opportunity_window=3)
        # Player 1, 2022: starts=25, appearances=30 → ratio=0.833.
        # Window for 2022 opp = [0.5, 0.75] → mean=0.625.
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2022)].iloc[0]
        assert row["starts_per_appearance_opp"] == pytest.approx(0.625, rel=1e-3)

    def test_missing_columns_silently_skipped(self) -> None:
        df = _toy_frame().drop(columns=["starts"])
        out = add_role_opportunity_features(df)
        # starts_opp and starts_per_appearance_opp must be absent
        assert "starts_opp" not in out.columns
        assert "starts_per_appearance_opp" not in out.columns
        # mins_played_opp must still be created
        assert "mins_played_opp" in out.columns

    def test_invalid_window_arguments_rejected(self) -> None:
        with pytest.raises(ValueError):
            RoleOpportunityFeatureTransformer(opportunity_window=0)
        with pytest.raises(ValueError):
            RoleOpportunityFeatureTransformer(recent_window=0)
        with pytest.raises(ValueError):
            RoleOpportunityFeatureTransformer(opportunity_window=2, recent_window=3)

    def test_get_feature_names_out(self) -> None:
        t = RoleOpportunityFeatureTransformer()
        names = t.get_feature_names_out()
        assert "mins_played_opp" in names
        assert "minutes_trend_opp" in names


class TestRollingSlopeHelper:
    def test_constant_series_returns_zero(self) -> None:
        s = pd.Series([100.0] * 5)
        out = _rolling_slope(s, window=3)
        # Each value uses a constant segment of length 1..3 → slope 0.
        assert (out.fillna(0.0) == 0.0).all()

    def test_strictly_increasing_returns_positive(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _rolling_slope(s, window=3)
        assert out.iloc[-1] > 0
        # Strictly increasing: last slope over [3,4,5] = 1.0 exactly.
        assert out.iloc[-1] == pytest.approx(1.0)

    def test_strictly_decreasing_returns_negative(self) -> None:
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        out = _rolling_slope(s, window=3)
        assert out.iloc[-1] < 0
        assert out.iloc[-1] == pytest.approx(-1.0)

    def test_nan_does_not_contaminate(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0, 4.0])
        out = _rolling_slope(s, window=3)
        # Last window = [nan, 3, 4] → valid subset is [3, 4] → slope 1.0.
        assert out.iloc[-1] == pytest.approx(1.0)
