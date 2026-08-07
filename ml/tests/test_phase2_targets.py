import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml.models.expected_minutes import ExpectedMinutesModel
from ml.targets.builder import TargetBuilder
from ml.targets.theoretical import TheoreticalFantavoto

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def base_df() -> pd.DataFrame:
    """Minimal synthetic player-season DataFrame."""
    np.random.seed(42)
    n = 40
    return pd.DataFrame(
        {
            "player_fotmob_id": [f"p{i % 10}" for i in range(n)],
            "season_start": [2022 + (i // 10) for i in range(n)],
            "canonical_role": ["GK", "DEF", "MID", "FWD"] * (n // 4),
            "appearances": np.random.randint(15, 35, n).astype(float),
            "mins_played": np.random.randint(1200, 3000, n).astype(float),
            "goals": np.random.randint(0, 20, n).astype(float),
            "goal_assist": np.random.randint(0, 10, n).astype(float),
            "clean_sheet": np.random.randint(0, 15, n).astype(float),
            "yellow_card": np.random.randint(0, 5, n).astype(float),
            "red_card": np.zeros(n),
            "own_goals": np.zeros(n),
            "penalty_scored": np.zeros(n),
            "penalty_missed": np.zeros(n),
            "fantavoto_medio": np.random.uniform(5.5, 7.5, n),
            "qt_a": np.random.randint(5, 80, n).astype(float),
            "team_strength_score": np.random.uniform(0.3, 0.9, n),
            "is_top_team": np.random.randint(0, 2, n),
            "role_code": [0, 1, 2, 3] * (n // 4),
            "season_idx": [2022 + (i // 10) - 2022 for i in range(n)],
        }
    )


@pytest.fixture
def em_training_df() -> pd.DataFrame:
    """Larger synthetic df for ExpectedMinutesModel training (4 seasons × 20 players)."""
    np.random.seed(0)
    rows = []
    for season in [2021, 2022, 2023, 2024]:
        for p in range(20):
            rows.append(
                {
                    "player_fotmob_id": f"p{p}",
                    "season_start": season,
                    "mins_played": float(np.random.randint(500, 3200)),
                    "appearances": float(np.random.randint(10, 38)),
                    "team_strength_score": float(np.random.uniform(0.3, 0.9)),
                    "is_top_team": int(np.random.randint(0, 2)),
                    "role_code": int(p % 4),
                    "season_idx": season - 2021,
                    "canonical_role": ["GK", "DEF", "MID", "FWD"][p % 4],
                }
            )
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTargetBuilder:
    def test_all_six_columns_produced(self, base_df):
        builder = TargetBuilder()
        result = builder.build(base_df)
        expected_cols = [
            "fantavoto_medio",
            "fantapunti_totali",
            "bonus_previsti",
            "minuti_giocati",
            "probabilita_titolarita",
            "prezzo_atteso",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing target column: {col}"

    def test_fantapunti_totali_formula(self, base_df):
        builder = TargetBuilder()
        result = builder.build(base_df)
        expected = result["fantavoto_medio"] * result["appearances"]
        pd.testing.assert_series_equal(
            result["fantapunti_totali"].round(6),
            expected.clip(lower=0.0).round(6),
            check_names=False,
        )

    def test_probabilita_titolarita_range(self, base_df):
        builder = TargetBuilder()
        result = builder.build(base_df)
        assert result["probabilita_titolarita"].between(0.0, 1.0).all(), (
            "probabilita_titolarita has values outside [0, 1]"
        )

    def test_prezzo_atteso_fallback_when_qt_a_absent(self, base_df):
        df_no_qt = base_df.drop(columns=["qt_a"])
        builder = TargetBuilder()
        result = builder.build(df_no_qt)
        assert (result["prezzo_atteso"] == 1.0).all()

    def test_bonus_previsti_increases_with_goals(self, base_df):
        builder = TargetBuilder()
        df_low = base_df.copy()
        df_high = base_df.copy()
        # All FWD for simplicity
        df_low["canonical_role"] = "FWD"
        df_high["canonical_role"] = "FWD"
        df_low["goals"] = 0.0
        df_high["goals"] = 20.0
        # Same everything else
        for col in ["goal_assist", "yellow_card", "red_card", "own_goals"]:
            df_low[col] = 0.0
            df_high[col] = 0.0

        res_low = builder.build(df_low)
        res_high = builder.build(df_high)
        assert res_high["bonus_previsti"].mean() > res_low["bonus_previsti"].mean()


class TestTheoreticalFantavoto:
    def test_output_range(self, base_df):
        feat = TheoreticalFantavoto()
        df_pl = pl.from_pandas(base_df)
        result = feat.safe_compute(df_pl)
        assert result.min() >= 1.0
        assert result.max() <= 10.0

    def test_fwd_default_no_crash(self):
        """Rows with no canonical_role should use FWD weights without crashing."""
        df = pl.DataFrame(
            {
                "goals_per90": [1.0, 0.5],
                "goal_assist_per90": [0.5, 0.2],
                "total_scoring_att_per90": [3.0, 2.0],
                "ontarget_scoring_att_per90": [2.0, 1.5],
                "won_contest_per90": [1.0, 0.8],
            }
        )
        # No canonical_role column
        feat = TheoreticalFantavoto()
        result = feat.safe_compute(df)
        assert len(result) == 2
        assert result.is_nan().sum() == 0

    def test_gk_scores_differently_than_fwd(self):
        """GK with saves and clean sheet should differ from FWD with same saves."""
        shared_vals = {
            "saves_per90": [2.0, 2.0],
            "goals_conceded_per90": [0.5, 0.5],
            "clean_sheet_per90": [0.5, 0.5],
            "_goals_prevented_per90": [1.0, 1.0],
            "goals_per90": [0.1, 0.1],
            "goal_assist_per90": [0.0, 0.0],
            "total_scoring_att_per90": [0.2, 0.2],
            "ontarget_scoring_att_per90": [0.1, 0.1],
            "won_contest_per90": [0.5, 0.5],
        }
        df = pl.DataFrame({**shared_vals, "canonical_role": ["GK", "FWD"]})
        feat = TheoreticalFantavoto()
        result = feat.safe_compute(df)
        # GK and FWD use different weights, so scores must differ
        assert result[0] != result[1], (
            "GK and FWD with same stats should score differently"
        )


class TestExpectedMinutesModel:
    def test_fit_raises_on_too_few_rows(self):
        tiny_df = pd.DataFrame(
            {
                "player_fotmob_id": ["p1"] * 5,
                "season_start": list(range(2020, 2025)),
                "mins_played": [1800.0] * 5,
                "appearances": [20.0] * 5,
                "role_code": [1] * 5,
                "season_idx": list(range(5)),
            }
        )
        model = ExpectedMinutesModel()
        with pytest.raises(ValueError, match="30"):
            model.fit(tiny_df)

    def test_predict_non_negative(self, em_training_df):
        model = ExpectedMinutesModel()
        model.fit(em_training_df)
        df_pred = em_training_df[em_training_df["season_start"] == 2024].copy()
        results = model.predict(df_pred)
        assert all(r.expected_minutes >= 0.0 for r in results)

    def test_backtest_uses_timeseries_split(self, em_training_df):
        """In each fold, max(train season) <= min(test season)."""
        df_feat = ExpectedMinutesModel.build_features(em_training_df)
        feature_cols = ExpectedMinutesModel._resolve_feature_cols(df_feat)
        df_sorted = df_feat.sort_values("season_start").copy()

        from sklearn.model_selection import TimeSeriesSplit

        X = df_sorted[feature_cols]
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X):
            train_seasons = df_sorted.iloc[train_idx]["season_start"]
            test_seasons = df_sorted.iloc[test_idx]["season_start"]
            assert train_seasons.max() <= test_seasons.min(), (
                f"TimeSeriesSplit leakage: train max {train_seasons.max()} "
                f"> test min {test_seasons.min()}"
            )

    def test_build_features_lag_correctness(self):
        """Season 2023's mins_played_lag1 must equal season 2022's mins_played."""
        df = pd.DataFrame(
            {
                "player_fotmob_id": ["p1", "p1", "p1"],
                "season_start": [2022, 2023, 2024],
                "mins_played": [2700.0, 1800.0, 2400.0],
                "appearances": [30.0, 20.0, 27.0],
            }
        )
        result = ExpectedMinutesModel.build_features(df)
        p1 = result[result["player_fotmob_id"] == "p1"].sort_values("season_start")
        assert p1.iloc[1]["mins_played_lag1"] == 2700.0, (
            "Season 2023 lag should be 2022 value"
        )
        assert p1.iloc[2]["mins_played_lag1"] == 1800.0, (
            "Season 2024 lag should be 2023 value"
        )
        assert pd.isna(p1.iloc[0]["mins_played_lag1"]), (
            "Season 2022 has no prior, should be NaN"
        )
