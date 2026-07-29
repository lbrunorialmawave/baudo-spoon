"""Tests for MANTRA ML feature integration.

Covers:
- MantraImputer fit/transform correctness
- Pipeline split safety (no leakage from test fold)
- Temporal lag correctness (no future data in features)
- Derived feature computation
"""

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.mantra_features import (
    MANTRA_FEATURE_COLS,
    MantraImputer,
    _compute_cumulative_lag,
    add_mantra_derived_features,
)


@pytest.fixture
def sample_df():
    """Multi-season, multi-role DataFrame for testing."""
    return pd.DataFrame({
        "player_fotmob_id": [1, 1, 1, 2, 2, 3],
        "season_start": [2021, 2022, 2023, 2022, 2023, 2023],
        "canonical_role": ["FWD", "FWD", "FWD", "MID", "MID", "GK"],
        "mantra_vote_avg": [6.5, 6.8, np.nan, 6.0, 6.2, np.nan],
        "mantra_vote_std": [0.5, 0.4, np.nan, 0.6, 0.5, np.nan],
        "mantra_minutes_avg": [2500, 2700, np.nan, 2000, 2100, np.nan],
        "mantra_xg_per90": [0.3, 0.35, np.nan, 0.1, 0.12, np.nan],
        "mantra_xa_per90": [0.1, 0.12, np.nan, 0.2, 0.22, np.nan],
        "mantra_presence_rate": [0.85, 0.9, np.nan, 0.7, 0.75, np.nan],
        "mantra_seasons_it": [1, 2, np.nan, 1, 2, np.nan],
    })


class TestMantraImputer:
    def test_fit_transform_no_nan_output(self, sample_df):
        imp = MantraImputer()
        imp.fit(sample_df)
        out = imp.transform(sample_df)
        for col in MANTRA_FEATURE_COLS:
            assert out[col].isna().sum() == 0, f"{col} still has NaN after imputation"

    def test_missing_flags_created(self, sample_df):
        imp = MantraImputer()
        out = imp.fit_transform(sample_df)
        for col in MANTRA_FEATURE_COLS:
            flag = f"{col}_missing"
            assert flag in out.columns
            # Player 1 season 2023 and player 3 should be flagged
            assert out[flag].sum() > 0

    def test_fit_transform_equals_fit_then_transform(self, sample_df):
        imp1 = MantraImputer()
        out1 = imp1.fit_transform(sample_df)

        imp2 = MantraImputer()
        imp2.fit(sample_df)
        out2 = imp2.transform(sample_df)

        pd.testing.assert_frame_equal(out1, out2)

    def test_pipeline_split_safety(self):
        """Verify imputer learns only from fit() data, not test data."""
        df_a = pd.DataFrame({
            "season_start": [2022] * 10,
            "canonical_role": ["FWD"] * 10,
            "mantra_vote_avg": [6.0] * 10,
            "mantra_vote_std": [0.5] * 10,
            "mantra_minutes_avg": [2000.0] * 10,
            "mantra_xg_per90": [0.2] * 10,
            "mantra_xa_per90": [0.1] * 10,
            "mantra_presence_rate": [0.8] * 10,
            "mantra_seasons_it": [2.0] * 10,
        })
        df_b = pd.DataFrame({
            "season_start": [2022] * 10,
            "canonical_role": ["FWD"] * 10,
            "mantra_vote_avg": [9.0] * 10,
            "mantra_vote_std": [0.1] * 10,
            "mantra_minutes_avg": [3000.0] * 10,
            "mantra_xg_per90": [0.8] * 10,
            "mantra_xa_per90": [0.5] * 10,
            "mantra_presence_rate": [0.95] * 10,
            "mantra_seasons_it": [5.0] * 10,
        })
        df_full = pd.concat([df_a, df_b], ignore_index=True)

        imp_fold = MantraImputer().fit(df_a)
        imp_full = MantraImputer().fit(df_full)

        mean_fold = imp_fold.group_means_["mantra_vote_avg"].loc[(2022, "FWD")]
        mean_full = imp_full.group_means_["mantra_vote_avg"].loc[(2022, "FWD")]

        assert np.isclose(mean_fold, 6.0), (
            f"Leakage: fold mean={mean_fold}, expected 6.0"
        )
        assert not np.isclose(mean_fold, mean_full), (
            "Test not discriminant: fold and full means are identical."
        )

    def test_unseen_role_uses_global_mean(self, sample_df):
        """Imputer handles unseen roles gracefully via global fallback."""
        imp = MantraImputer().fit(sample_df)
        test_df = pd.DataFrame({
            "season_start": [2024],
            "canonical_role": ["UNKNOWN_ROLE"],
            "mantra_vote_avg": [np.nan],
            "mantra_vote_std": [np.nan],
            "mantra_minutes_avg": [np.nan],
            "mantra_xg_per90": [np.nan],
            "mantra_xa_per90": [np.nan],
            "mantra_presence_rate": [np.nan],
            "mantra_seasons_it": [np.nan],
        })
        out = imp.transform(test_df)
        # Should get global mean, not crash
        assert out["mantra_vote_avg"].notna().all()


class TestCumulativeLag:
    def test_no_leakage(self):
        """Cumulative lag must use only prior seasons, never current."""
        raw = pd.DataFrame({
            "player_fotmob_id": [1, 1, 1],
            "season_start": [2021, 2022, 2023],
            "vote_avg": [6.0, 7.0, 8.0],
            "vote_std": [0.5, 0.4, 0.3],
            "minutes_avg": [2000, 2500, 3000],
            "xg_per90": [0.2, 0.3, 0.4],
            "xa_per90": [0.1, 0.15, 0.2],
            "presence_rate": [0.7, 0.8, 0.9],
        })
        result = _compute_cumulative_lag(raw)

        # Season 2021: no prior data → NaN
        row_2021 = result[result["season_start"] == 2021].iloc[0]
        assert pd.isna(row_2021["mantra_vote_avg"])
        assert row_2021["mantra_seasons_it"] == 0

        # Season 2022: only 2021 data
        row_2022 = result[result["season_start"] == 2022].iloc[0]
        assert np.isclose(row_2022["mantra_vote_avg"], 6.0)
        assert row_2022["mantra_seasons_it"] == 1

        # Season 2023: avg of 2021 + 2022
        row_2023 = result[result["season_start"] == 2023].iloc[0]
        assert np.isclose(row_2023["mantra_vote_avg"], 6.5)
        assert row_2023["mantra_seasons_it"] == 2

    def test_current_season_never_included(self):
        """The feature for season N must NOT include season N's own data."""
        raw = pd.DataFrame({
            "player_fotmob_id": [1, 1],
            "season_start": [2022, 2023],
            "vote_avg": [5.0, 9.0],
            "vote_std": [1.0, 0.1],
            "minutes_avg": [1000, 3000],
            "xg_per90": [0.1, 0.9],
            "xa_per90": [0.05, 0.5],
            "presence_rate": [0.5, 0.99],
        })
        result = _compute_cumulative_lag(raw)

        # Season 2023 should only see 2022's data (5.0), NOT 9.0
        row_2023 = result[result["season_start"] == 2023].iloc[0]
        assert np.isclose(row_2023["mantra_vote_avg"], 5.0)
        assert not np.isclose(row_2023["mantra_vote_avg"], 9.0)


class TestDerivedFeatures:
    def test_voto_trend_above_average(self):
        df = pd.DataFrame({
            "mantra_vote_avg": [7.0, 6.0, 5.0],
            "mantra_vote_std": [0.3, 0.5, 0.8],
            "mantra_xg_per90": [0.4, 0.2, 0.1],
            "canonical_role": ["FWD", "FWD", "FWD"],
            "goals_per90": [0.3, 0.2, 0.1],
        })
        out = add_mantra_derived_features(df)

        # Player with highest vote should have trend > 1
        assert out.loc[0, "mantra_voto_trend"] > 1.0
        # Player with lowest vote should have trend < 1
        assert out.loc[2, "mantra_voto_trend"] < 1.0

    def test_consistency_higher_for_stable_player(self):
        df = pd.DataFrame({
            "mantra_vote_avg": [7.0, 7.0],
            "mantra_vote_std": [0.2, 1.5],
            "mantra_xg_per90": [0.3, 0.3],
            "canonical_role": ["FWD", "FWD"],
            "goals_per90": [0.3, 0.3],
        })
        out = add_mantra_derived_features(df)
        # Lower std → higher consistency
        assert out.loc[0, "mantra_consistency"] > out.loc[1, "mantra_consistency"]

    def test_expected_ratio_capped_at_5(self):
        df = pd.DataFrame({
            "mantra_vote_avg": [6.5],
            "mantra_vote_std": [0.5],
            "mantra_xg_per90": [0.5],
            "canonical_role": ["FWD"],
            "goals_per90": [0.0],  # denominator near zero → ratio capped
        })
        out = add_mantra_derived_features(df)
        assert out.loc[0, "mantra_expected_ratio"] <= 5.0
