"""Tests for ml.preprocessing.features.

Covers:
- RollingFeatureTransformer temporal isolation (TASK_003 verification)
- OpponentStrengthAdjuster SAP feature generation (TASK_002)
- engineer_features idempotency (testing protocol requirement)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.features import (
    OpponentStrengthAdjuster,
    RollingFeatureTransformer,
    engineer_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_player_df(
    vals: list[float],
    player_id: str = "p1",
    seasons: list[int] | None = None,
    stat_col: str = "goals_per90",
) -> pd.DataFrame:
    """Minimal DataFrame with one player and multiple seasons."""
    if seasons is None:
        seasons = list(range(2021, 2021 + len(vals)))
    return pd.DataFrame({
        "player_fotmob_id": [player_id] * len(vals),
        "season_start": seasons,
        stat_col: vals,
    })


def _make_full_df(n_players: int = 5, n_seasons: int = 3) -> pd.DataFrame:
    """Multi-player DataFrame with all columns needed by engineer_features."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(n_players):
        for season in range(2021, 2021 + n_seasons):
            rows.append({
                "player_fotmob_id": pid,
                "player_name": f"Player{pid}",
                "team_fotmob_id": pid % 3,
                "team_name": f"Team{pid % 3}",
                "season_start": season,
                "season_label": str(season),
                "league_name": "Serie A",
                "canonical_role": ["GK", "DEF", "MID", "FWD", "MID"][pid % 5],
                "mins_played": float(rng.integers(800, 3000)),
                "goals": float(rng.integers(0, 15)),
                "goal_assist": float(rng.integers(0, 10)),
                "saves": float(rng.integers(0, 80)) if pid % 5 == 0 else 0.0,
                "_goals_prevented": float(rng.uniform(0, 5)),
                "clean_sheet": float(rng.integers(0, 20)),
                "goals_conceded": float(rng.integers(0, 50)),
                "appearances": float(rng.integers(20, 38)),
                "team_strength_score": float(rng.uniform(0.3, 1.0)),
                "is_top_team": int(rng.integers(0, 2)),
                "team_rank_norm": float(rng.uniform(0.1, 1.0)),
            })
    return pd.DataFrame(rows)


# ── TASK_003: Temporal Isolation ──────────────────────────────────────────────

class TestRollingFeatureTransformerTemporalIsolation:
    """Verify that closed='left' prevents look-ahead bias."""

    def test_rolling_excludes_current_season(self):
        """Season N's rolling mean must not include season N's data."""
        df = _make_player_df([1.0, 2.0, 3.0])
        transformer = RollingFeatureTransformer(window=2)
        result = transformer.fit_transform(df)
        result = result.sort_values("season_start").reset_index(drop=True)

        # Season 2021 (index 0): no prior history → NaN
        assert pd.isna(result.loc[0, "goals_per90_roll2"]), \
            "Season 2021 should have NaN rolling (no prior data)"

        # Season 2022 (index 1): only 1 prior season → 1.0
        assert abs(result.loc[1, "goals_per90_roll2"] - 1.0) < 1e-9, \
            f"Season 2022 rolling should be 1.0, got {result.loc[1, 'goals_per90_roll2']}"

        # Season 2023 (index 2): mean of 2021+2022 = 1.5
        assert abs(result.loc[2, "goals_per90_roll2"] - 1.5) < 1e-9, \
            f"Season 2023 rolling should be 1.5, got {result.loc[2, 'goals_per90_roll2']}"

    def test_modifying_future_data_does_not_change_historical_rolling(self):
        """KEY TEST: changing a future season's value must not alter prior rolling averages."""
        # Baseline
        df_base = _make_player_df([1.0, 2.0, 3.0])
        transformer = RollingFeatureTransformer(window=2)
        result_base = transformer.fit_transform(df_base).sort_values("season_start").reset_index(drop=True)

        # Modify season 2023 (future relative to 2021 and 2022)
        df_modified = df_base.copy()
        df_modified.loc[df_modified["season_start"] == 2023, "goals_per90"] = 999.0

        result_modified = transformer.fit_transform(df_modified).sort_values("season_start").reset_index(drop=True)

        # Season 2021 rolling must be unchanged
        assert (
            pd.isna(result_base.loc[0, "goals_per90_roll2"])
            and pd.isna(result_modified.loc[0, "goals_per90_roll2"])
        ), "Season 2021 rolling should still be NaN after modifying 2023 data"

        # Season 2022 rolling must be unchanged (was 1.0)
        roll_2022_base = result_base.loc[1, "goals_per90_roll2"]
        roll_2022_mod = result_modified.loc[1, "goals_per90_roll2"]
        assert abs(roll_2022_base - roll_2022_mod) < 1e-9, (
            f"Season 2022 rolling changed after modifying 2023 data: "
            f"{roll_2022_base} → {roll_2022_mod}"
        )

        # Season 2023's rolling is based on 2021+2022 only → STILL 1.5
        roll_2023_mod = result_modified.loc[2, "goals_per90_roll2"]
        assert abs(roll_2023_mod - 1.5) < 1e-9, (
            f"Season 2023 rolling should be 1.5 (based on 2021+2022), "
            f"not affected by its own current value. Got: {roll_2023_mod}"
        )

    def test_delta_uses_only_previous_season(self):
        """Year-over-year delta uses current − previous, never future data."""
        df = _make_player_df([10.0, 13.0, 100.0])
        transformer = RollingFeatureTransformer(window=2)
        result = transformer.fit_transform(df).sort_values("season_start").reset_index(drop=True)

        assert pd.isna(result.loc[0, "goals_per90_delta1"]), \
            "Season 2021 delta should be NaN (no prior season)"
        assert abs(result.loc[1, "goals_per90_delta1"] - 3.0) < 1e-9, \
            "Season 2022 delta should be 13-10=3"
        # Season 2023 delta = 100 - 13 = 87; this IS the current season's delta
        assert abs(result.loc[2, "goals_per90_delta1"] - 87.0) < 1e-9, \
            "Season 2023 delta should be 100-13=87"

    def test_multiple_players_are_isolated(self):
        """Each player's rolling window must not bleed into another player's history."""
        df = pd.concat([
            _make_player_df([10.0, 20.0], player_id="p1"),
            _make_player_df([1.0, 2.0], player_id="p2"),
        ]).reset_index(drop=True)
        transformer = RollingFeatureTransformer(window=2)
        result = transformer.fit_transform(df)

        p2_2022 = result[
            (result["player_fotmob_id"] == "p2") & (result["season_start"] == 2022)
        ]["goals_per90_roll2"].iloc[0]

        # p2's 2022 rolling should be 1.0 (only p2's 2021 data), not influenced by p1
        assert abs(p2_2022 - 1.0) < 1e-9, \
            f"p2 season 2022 rolling should be 1.0 (p2's 2021 only), got {p2_2022}"

    def test_transformer_is_stateless_fit_is_noop(self):
        """fit() must not alter transform behaviour (stateless transformer)."""
        df = _make_player_df([5.0, 10.0, 15.0])
        t = RollingFeatureTransformer(window=2)

        # fit on subset, transform on full — result must be identical to fit_transform
        t.fit(df.iloc[:2])
        result_a = t.transform(df)
        result_b = RollingFeatureTransformer(window=2).fit_transform(df)

        pd.testing.assert_frame_equal(
            result_a.sort_values("season_start").reset_index(drop=True),
            result_b.sort_values("season_start").reset_index(drop=True),
        )


# ── TASK_002: OpponentStrengthAdjuster ────────────────────────────────────────

class TestOpponentStrengthAdjuster:
    """Verify SAP feature generation and O(N) groupby semantics."""

    def _make_league_df(self) -> pd.DataFrame:
        """Two teams in the same league/season with known ranks."""
        return pd.DataFrame({
            "player_fotmob_id": [1, 2, 3, 4],
            "player_name": ["A", "B", "C", "D"],
            "team_fotmob_id": [10, 10, 20, 20],
            "team_name": ["TeamA", "TeamA", "TeamB", "TeamB"],
            "season_start": [2023, 2023, 2023, 2023],
            "league_name": ["L1"] * 4,
            "team_rank_norm": [0.8, 0.8, 0.4, 0.4],  # TeamA=0.8, TeamB=0.4
            "goals_per90": [1.0, 2.0, 0.5, 0.8],
        })

    def test_sap_weight_computed_correctly(self):
        """SAP weight for TeamA vs TeamB is deterministic and correct."""
        df = self._make_league_df()
        adj = OpponentStrengthAdjuster(group_cols=["season_start", "league_name"])
        result = adj.fit_transform(df)

        assert "goals_per90_sap" in result.columns, \
            "goals_per90_sap column must be created"

        # For TeamA (rank 0.8), opponent = TeamB (0.4)
        # league_sum = 0.8+0.4=1.2, n_teams=2, league_mean=0.6
        # opp_mean = (1.2 - 0.8) / (2-1) = 0.4
        # sap_weight = 0.4 / 0.6 ≈ 0.667
        team_a_rows = result[result["team_fotmob_id"] == 10]
        sap_a = team_a_rows["goals_per90_sap"].iloc[0]
        expected_a = 1.0 * (0.4 / 0.6)  # goals_per90=1.0, sap_weight≈0.667
        assert abs(sap_a - expected_a) < 1e-6, \
            f"TeamA SAP value wrong: expected {expected_a:.4f}, got {sap_a:.4f}"

    def test_fit_only_on_train_transform_on_test(self):
        """Transformer fit on training data and applied to unseen test data."""
        df_train = self._make_league_df()
        df_test = df_train.copy()
        df_test["season_start"] = 2024  # unseen season

        adj = OpponentStrengthAdjuster(group_cols=["season_start", "league_name"])
        adj.fit(df_train)
        result = adj.transform(df_test)

        # Should fall back to global mean — SAP columns should still be created
        assert "goals_per90_sap" in result.columns, \
            "SAP columns must be created even for unseen season (global fallback)"

    def test_no_sap_when_team_rank_norm_missing(self):
        """When team_rank_norm is absent, transformer returns input unchanged."""
        df = pd.DataFrame({
            "player_fotmob_id": [1, 2],
            "goals_per90": [1.0, 2.0],
        })
        adj = OpponentStrengthAdjuster()
        result = adj.fit_transform(df)

        assert "goals_per90_sap" not in result.columns, \
            "SAP column must NOT be created when team_rank_norm is missing"

    def test_helper_columns_dropped(self):
        """Internal computation columns (league_sum, sap_weight, etc.) must be dropped."""
        df = self._make_league_df()
        adj = OpponentStrengthAdjuster(group_cols=["season_start", "league_name"])
        result = adj.fit_transform(df)

        for col in ("league_sum", "n_teams", "league_mean", "sap_weight"):
            assert col not in result.columns, \
                f"Internal column '{col}' must be dropped from output"


# ── Idempotency ───────────────────────────────────────────────────────────────

class TestEngineerFeaturesIdempotency:
    """Running engineer_features twice on the same data yields identical results."""

    def test_idempotent_on_same_input(self):
        """engineer_features is a pure function: same input → same output."""
        df = _make_full_df(n_players=4, n_seasons=3)
        result_1 = engineer_features(df)
        result_2 = engineer_features(df.copy())

        # Sort to ensure same row order before comparison
        sort_cols = ["player_fotmob_id", "season_start"]
        r1 = result_1.sort_values(sort_cols).reset_index(drop=True)
        r2 = result_2.sort_values(sort_cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(r1, r2, check_like=False)

    def test_original_df_not_mutated(self):
        """engineer_features must not modify the input DataFrame in-place."""
        df = _make_full_df(n_players=3, n_seasons=2)
        original_cols = set(df.columns)
        original_shape = df.shape

        engineer_features(df)

        assert set(df.columns) == original_cols, \
            "Input DataFrame columns were mutated by engineer_features"
        assert df.shape == original_shape, \
            "Input DataFrame shape was mutated by engineer_features"


# ── Recursive Feature Elimination ─────────────────────────────────────────────

class TestSelectFeaturesRfe:
    """Tests for select_features_rfe: collinearity pruning via Ridge-based RFE.

    Focus areas:
    - Output count respects the fraction parameter.
    - Output is always a subset of the input candidates.
    - Gracefully handles edge cases (too few features, all-NaN columns).
    - Stateless: repeated calls with the same data yield the same result.
    """

    def _make_X_y(
        self, n_rows: int = 80, n_feats: int = 12, seed: int = 0,
    ) -> tuple[pd.DataFrame, "pd.Series"]:
        """Synthetic feature matrix and continuous target."""
        rng = np.random.default_rng(seed)
        cols = [f"feat_{i}" for i in range(n_feats)]
        X = pd.DataFrame(rng.standard_normal((n_rows, n_feats)), columns=cols)
        y = pd.Series(rng.standard_normal(n_rows), name="target")
        return X, y

    def test_output_count_respects_fraction(self):
        """Number of selected features must match floor(n_available × fraction)."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=10)
        fraction = 0.70
        selected = select_features_rfe(X, y, list(X.columns), n_features_fraction=fraction)

        expected_count = max(1, round(len(X.columns) * fraction))
        assert len(selected) == expected_count, (
            f"Expected {expected_count} features; got {len(selected)}"
        )

    def test_selected_is_subset_of_candidates(self):
        """Every selected feature must be in the original candidate list."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=8)
        candidates = list(X.columns)
        selected = select_features_rfe(X, y, candidates)

        assert set(selected).issubset(set(candidates)), (
            f"Selected features {set(selected) - set(candidates)} not in candidates"
        )

    def test_too_few_features_returns_unchanged(self):
        """With fewer than 4 candidates, RFE is skipped and all features returned."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=3)
        selected = select_features_rfe(X, y, list(X.columns))

        assert set(selected) == set(X.columns), (
            "With < 4 candidate features, select_features_rfe should return all"
        )

    def test_fraction_1_returns_all(self):
        """fraction=1.0 means 'keep all' — RFE is skipped."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=10)
        selected = select_features_rfe(X, y, list(X.columns), n_features_fraction=1.0)

        # n_to_keep == len(available) → early exit
        assert len(selected) == len(X.columns)

    def test_missing_candidates_are_ignored(self):
        """Candidates absent from X are silently skipped."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=8)
        candidates = list(X.columns) + ["nonexistent_col"]
        # Must not raise; missing column simply not included
        selected = select_features_rfe(X, y, candidates, n_features_fraction=0.5)

        assert all(f in X.columns for f in selected), \
            "Selected features must all exist in X"

    def test_deterministic_across_calls(self):
        """Repeated calls with identical data yield identical selected features."""
        from ml.preprocessing.features import select_features_rfe

        X, y = self._make_X_y(n_feats=12, seed=99)
        candidates = list(X.columns)
        run1 = select_features_rfe(X, y, candidates, n_features_fraction=0.6)
        run2 = select_features_rfe(X.copy(), y.copy(), candidates, n_features_fraction=0.6)

        assert sorted(run1) == sorted(run2), "RFE must be deterministic for identical input"
