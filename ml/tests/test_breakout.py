"""Unit tests for the breakout dataset module (PR6)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.breakout import (
    DEFAULT_BREAKOUT_TARGET_MINUTES,
    BreakoutDatasetStats,
    build_breakout_dataset,
    build_breakout_labels,
    engineer_breakout_features,
)
from ml.sample_reliability.cohort import COHORT_LIMITED, COHORT_STANDARD


def _toy_frame() -> pd.DataFrame:
    """2 players, 4 seasons, one explicit breakout case.

    Player 1: LIMITED (300 min) in 2020 → STANDARD (1200) in 2021 → breakout.
    Player 1: STANDARD in 2021 → no label (only LIMITED rows get labels).
    Player 1: STANDARD in 2022 → no label.
    Player 1: STANDARD in 2023 → no label (and no next season either).

    Player 2: LIMITED (200 min) in 2020 → LIMITED (250) in 2021 → no breakout.
    Player 2: LIMITED (250) in 2021 → LIMITED (300) in 2022 → no breakout.
    Player 2: LIMITED (300) in 2022 → STANDARD (1000) in 2023 → breakout.
    Player 2: STANDARD (1000) in 2023 → no next season → no label.
    """
    return pd.DataFrame(
        {
            "player_fotmob_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "season_start": [
                2020, 2021, 2022, 2023,
                2020, 2021, 2022, 2023,
            ],
            "mins_played": [
                300, 1200, 1500, 1700,  # player 1
                200, 250, 300, 1000,   # player 2
            ],
            "starts": [3, 30, 35, 36, 2, 3, 4, 28],
            "appearances": [10, 33, 36, 37, 8, 9, 12, 32],
        }
    )


class TestEngineerBreakoutFeatures:
    def test_lag1_uses_previous_season(self) -> None:
        df = engineer_breakout_features(_toy_frame())
        # Player 1, 2020: no previous season → lag1 = NaN.
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2020)].iloc[0]
        assert math.isnan(row["mins_played_lag1"])
        # Player 1, 2021: lag1 = 300 (2020 minutes).
        row = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2021)].iloc[0]
        assert row["mins_played_lag1"] == 300.0

    def test_other_columns_untouched(self) -> None:
        df = engineer_breakout_features(_toy_frame())
        assert "starts" in df.columns
        assert "appearances" in df.columns
        # Lagged versions added
        assert "starts_lag1" in df.columns
        assert "appearances_lag1" in df.columns


class TestBuildBreakoutLabels:
    def test_label_values_match_spec(self) -> None:
        df = _toy_frame()
        labels = build_breakout_labels(df)
        # Player 1, 2020: LIMITED(300) → STANDARD(1200) at 2021 → label=1.
        idx_1_2020 = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2020)].index[0]
        assert labels.loc[idx_1_2020] == 1

        # Player 2, 2020: LIMITED(200) → LIMITED(250) at 2021 → label=0.
        idx_2_2020 = df[(df["player_fotmob_id"] == 2) & (df["season_start"] == 2020)].index[0]
        assert labels.loc[idx_2_2020] == 0

        # Player 2, 2022: LIMITED(300) → STANDARD(1000) at 2023 → label=1.
        idx_2_2022 = df[(df["player_fotmob_id"] == 2) & (df["season_start"] == 2022)].index[0]
        assert labels.loc[idx_2_2022] == 1

        # Player 1, 2021: STANDARD(1200) → no label.
        idx_1_2021 = df[(df["player_fotmob_id"] == 1) & (df["season_start"] == 2021)].index[0]
        assert math.isnan(labels.loc[idx_1_2021])

    def test_no_next_season_means_nan(self) -> None:
        df = _toy_frame()
        labels = build_breakout_labels(df)
        # Player 1, 2023: STANDARD, no label anyway.
        # Player 2, 2023: STANDARD, no label anyway.
        # To test "no next season" semantics, we need a LIMITED row whose
        # next season doesn't exist.  We construct that case explicitly.
        df2 = pd.DataFrame(
            {
                "player_fotmob_id": [99, 99],
                "season_start": [2020, 2023],
                "mins_played": [200, 1500],  # both LIMITED... wait
            }
        )
        # Actually 200 is LIMITED, 1500 is STANDARD, so 2023 has no label
        # because it's not LIMITED.  We need a LIMITED row without a next season.
        df2 = pd.DataFrame(
            {
                "player_fotmob_id": [99],
                "season_start": [2023],
                "mins_played": [200],  # LIMITED, but no 2024 row.
            }
        )
        labels2 = build_breakout_labels(df2)
        assert math.isnan(labels2.iloc[0])

    def test_missing_minutes_column_raises(self) -> None:
        with pytest.raises(KeyError):
            build_breakout_labels(pd.DataFrame({"foo": [1, 2]}))


class TestBuildBreakoutDataset:
    def test_returns_features_labels_and_stats(self) -> None:
        df = _toy_frame()
        features, labels, stats = build_breakout_dataset(
            df, feature_columns=["mins_played", "starts", "appearances"],
        )
        # Only LIMITED rows with a next season appear in the dataset.
        # Player 1, 2020: yes (LIMITED + next = 2021)
        # Player 2, 2020: yes (LIMITED + next = 2021)
        # Player 2, 2021: yes (LIMITED + next = 2022)
        # Player 2, 2022: yes (LIMITED + next = 2023)
        # Total: 4 rows
        assert stats.n_total == 4
        assert stats.n_positive == 2  # p1-2020, p2-2022
        assert stats.n_negative == 2  # p2-2020, p2-2021
        assert math.isclose(stats.base_rate, 0.5)
        # STANDARD rows excluded.
        excluded = 8 - 4
        assert stats.n_excluded_no_next_season == excluded

    def test_feature_whitelist(self) -> None:
        df = _toy_frame()
        features, labels, _ = build_breakout_dataset(
            df, feature_columns=["starts"],
        )
        assert list(features.columns) == ["starts"]
        assert len(features) == 4

    def test_default_features_excludes_identifiers(self) -> None:
        df = _toy_frame()
        features, labels, _ = build_breakout_dataset(df)
        # player_fotmob_id, season_start, mins_played must be excluded
        assert "player_fotmob_id" not in features.columns
        assert "season_start" not in features.columns
        assert "mins_played" not in features.columns
        assert "starts" in features.columns

    def test_stats_is_dataclass(self) -> None:
        df = _toy_frame()
        _, _, stats = build_breakout_dataset(df)
        assert isinstance(stats, BreakoutDatasetStats)
        assert stats.target_minutes == DEFAULT_BREAKOUT_TARGET_MINUTES


# ── Leakage tests (plan §45) ────────────────────────────────────────────────


class TestLeakageInvariants:
    def test_label_uses_strictly_future_minutes(self) -> None:
        """The label at season t must depend *only* on the minutes at
        season ``t+1`` and the cohort at season ``t``.

        We verify this by keeping the cohort membership at ``t`` fixed
        and flipping the *future* minutes: the label must change in the
        opposite direction.
        """
        df = _toy_frame()
        labels_a = build_breakout_labels(df).dropna()

        # Take the labelled rows for player 2 and shift the future
        # minutes to a value that flips the label from 0 to 1.
        df2 = df.copy()
        # Player 2, 2021: LIMITED(250) → LIMITED(250) at 2022 → label 0.
        # Make the 2022 row standard → label at 2021 becomes 1.
        df2.loc[
            (df2["player_fotmob_id"] == 2) & (df2["season_start"] == 2022),
            "mins_played",
        ] = 1200  # STANDARD at t+1
        labels_b = build_breakout_labels(df2).dropna()

        # The label for player 2 at season 2021 must have flipped 0 → 1.
        idx = df[(df["player_fotmob_id"] == 2) & (df["season_start"] == 2021)].index[0]
        assert labels_a.loc[idx] == 0
        assert labels_b.loc[idx] == 1

    def test_engineered_features_use_strictly_lagged_values(self) -> None:
        """After engineering, the lag1 column at row ``t`` must match
        the raw value of the same player at ``t-1``, never at ``t``."""
        df = engineer_breakout_features(_toy_frame())
        # Build a direct comparison frame.
        original = _toy_frame().set_index(["player_fotmob_id", "season_start"])
        for _, row in df.iterrows():
            pid = int(row["player_fotmob_id"])
            season = int(row["season_start"])
            lag = row["mins_played_lag1"]
            if pd.isna(lag):
                continue
            raw_prev = original.loc[(pid, season - 1), "mins_played"]
            assert lag == float(raw_prev)
            # And the lag must NOT match the current season's minutes
            raw_curr = original.loc[(pid, season), "mins_played"]
            assert lag != float(raw_curr)
