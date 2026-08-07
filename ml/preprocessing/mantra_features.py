"""MANTRA historical features for ML training.

Attaches lagged MANTRA stats (vote_avg, vote_std, presence_rate, etc.)
to the player-season DataFrame. All features use strictly historical data
(seasons < current) to guarantee zero temporal leakage.

Features produced:
    mantra_vote_avg       — cumulative mean vote from prior seasons
    mantra_vote_std       — cumulative vote std dev from prior seasons
    mantra_minutes_avg    — cumulative avg minutes/season
    mantra_xg_per90       — cumulative expected goals per 90'
    mantra_xa_per90       — cumulative expected assists per 90'
    mantra_presence_rate  — cumulative presence rate (0-1)
    mantra_seasons_it     — number of prior Serie A seasons

Derived features (Fase 2):
    mantra_voto_trend       — vote_avg / role_median_vote (>1 = above average)
    mantra_consistency      — vote_avg / (vote_std + 0.1)
    mantra_expected_ratio   — xg_per90 / (goals_per90 + 0.01), capped at 5
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger(__name__)

MANTRA_FEATURE_COLS: list[str] = [
    "mantra_vote_avg",
    "mantra_vote_std",
    "mantra_minutes_avg",
    "mantra_xg_per90",
    "mantra_xa_per90",
    "mantra_presence_rate",
    "mantra_seasons_it",
]

MANTRA_DERIVED_COLS: list[str] = [
    "mantra_voto_trend",
    "mantra_consistency",
    "mantra_expected_ratio",
]

_MANTRA_ML_SQL = sa.text("""
    SELECT
        player_fotmob_id,
        season_start,
        mantra_vote_avg,
        mantra_vote_std,
        mantra_minutes_avg,
        mantra_xg_per90,
        mantra_xa_per90,
        mantra_presence_rate,
        mantra_seasons_it
    FROM player_mantra_ml_features
""")

# Fallback: compute MANTRA features directly from player_season_stats
# using a lag-1 self-join in pandas (when the view doesn't exist).
_FALLBACK_STATS_SQL = sa.text("""
    SELECT
        pss.player_fotmob_id,
        s.season_start,
        AVG(CASE WHEN pss.stat_category = 'rating' THEN pss.value END) AS vote_avg,
        STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END) AS vote_std,
        AVG(CASE WHEN pss.stat_category = 'mins_played' THEN pss.value END) AS minutes_avg,
        AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90' THEN pss.value END) AS xg_per90,
        AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90' THEN pss.value END) AS xa_per90,
        LEAST(
            AVG(CASE WHEN pss.stat_category = 'mins_played' THEN pss.value END) / 3420.0,
            1.0
        ) AS presence_rate
    FROM player_season_stats pss
    JOIN seasons s ON s.id = pss.season_id
    GROUP BY pss.player_fotmob_id, s.season_start
    ORDER BY pss.player_fotmob_id, s.season_start
""")


def _compute_cumulative_lag(df_per_season: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative averages from strictly prior seasons (pandas fallback)."""
    df = df_per_season.sort_values(["player_fotmob_id", "season_start"])
    result_rows = []

    for pid, grp in df.groupby("player_fotmob_id"):
        grp = grp.sort_values("season_start").reset_index(drop=True)
        for i in range(len(grp)):
            row = {"player_fotmob_id": pid, "season_start": grp.loc[i, "season_start"]}
            prior = grp.iloc[:i]
            if prior.empty:
                for col in [
                    "vote_avg",
                    "vote_std",
                    "minutes_avg",
                    "xg_per90",
                    "xa_per90",
                    "presence_rate",
                ]:
                    row[f"mantra_{col}"] = np.nan
                row["mantra_seasons_it"] = 0
            else:
                row["mantra_vote_avg"] = prior["vote_avg"].mean()
                row["mantra_vote_std"] = prior["vote_std"].mean()
                row["mantra_minutes_avg"] = prior["minutes_avg"].mean()
                row["mantra_xg_per90"] = prior["xg_per90"].mean()
                row["mantra_xa_per90"] = prior["xa_per90"].mean()
                row["mantra_presence_rate"] = prior["presence_rate"].mean()
                row["mantra_seasons_it"] = len(prior)
            result_rows.append(row)

    return pd.DataFrame(result_rows)


def attach_mantra_features(
    df: pd.DataFrame,
    engine: sa.Engine,
) -> pd.DataFrame:
    """Attach MANTRA ML features to the player-season DataFrame.

    Tries the DB view first; falls back to computing from raw stats.
    All features are strictly lagged (use only prior seasons).
    """
    if not {"player_fotmob_id", "season_start"}.issubset(df.columns):
        raise ValueError("df must contain 'player_fotmob_id' and 'season_start'.")

    mantra_df: pd.DataFrame | None = None

    # Try the dedicated view
    try:
        mantra_df = pd.read_sql(_MANTRA_ML_SQL, engine)
        if mantra_df.empty:
            mantra_df = None
            log.info("player_mantra_ml_features view is empty, using fallback.")
    except sa.exc.ProgrammingError:
        log.info(
            "player_mantra_ml_features view not found — using pandas fallback. "
            "Apply migration 016_add_mantra_ml_features_view.sql for better performance."
        )
    except Exception as exc:
        log.warning("Could not read MANTRA ML view (%s); using fallback.", exc)

    # Fallback: compute from raw stats
    if mantra_df is None:
        try:
            raw_stats = pd.read_sql(_FALLBACK_STATS_SQL, engine)
            if raw_stats.empty:
                log.warning(
                    "No player_season_stats for MANTRA features. All will be NaN."
                )
                for col in MANTRA_FEATURE_COLS:
                    df[col] = np.nan
                return df
            mantra_df = _compute_cumulative_lag(raw_stats)
        except Exception as exc:
            log.warning("MANTRA fallback failed (%s). Features will be NaN.", exc)
            for col in MANTRA_FEATURE_COLS:
                df[col] = np.nan
            return df

    # Merge
    df = df.merge(
        mantra_df[["player_fotmob_id", "season_start"] + MANTRA_FEATURE_COLS],
        on=["player_fotmob_id", "season_start"],
        how="left",
    )

    n_with = int(df["mantra_vote_avg"].notna().sum())
    log.info(
        "MANTRA features: %d / %d rows have historical data (%.1f%%).",
        n_with,
        len(df),
        100.0 * n_with / len(df) if len(df) else 0,
    )
    return df


def add_mantra_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived MANTRA features (Fase 2 of the plan).

    Only computed when base MANTRA features exist.
    """
    df = df.copy()

    # mantra_voto_trend: vote_avg / role_median_vote
    if "mantra_vote_avg" in df.columns and "canonical_role" in df.columns:
        role_median = df.groupby("canonical_role")["mantra_vote_avg"].transform(
            "median"
        )
        df["mantra_voto_trend"] = df["mantra_vote_avg"] / role_median.clip(lower=0.1)
    else:
        df["mantra_voto_trend"] = np.nan

    # mantra_consistency: vote_avg / (vote_std + 0.1)
    if "mantra_vote_avg" in df.columns and "mantra_vote_std" in df.columns:
        df["mantra_consistency"] = df["mantra_vote_avg"] / (
            df["mantra_vote_std"].fillna(0) + 0.1
        )
    else:
        df["mantra_consistency"] = np.nan

    # mantra_expected_ratio: xg_per90 / (goals_per90 + 0.01), cap at 5
    if "mantra_xg_per90" in df.columns:
        goals_col = "goals_per90" if "goals_per90" in df.columns else None
        if goals_col:
            denom = pd.to_numeric(df[goals_col], errors="coerce").fillna(0) + 0.01
            df["mantra_expected_ratio"] = (df["mantra_xg_per90"] / denom).clip(
                upper=5.0
            )
        else:
            df["mantra_expected_ratio"] = np.nan
    else:
        df["mantra_expected_ratio"] = np.nan

    return df


class MantraImputer(BaseEstimator, TransformerMixin):
    """Impute missing MANTRA features with hierarchical fallback.

    Fallback order: mean(season, role) -> mean(role) -> global mean.
    All statistics learned ONLY from fit() data (training fold).

    MUST be used inside a sklearn Pipeline with fit() called exclusively
    on the training fold of each split.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        role_col: str = "canonical_role",
        season_col: str = "season_start",
    ):
        self.feature_cols = feature_cols or MANTRA_FEATURE_COLS
        self.role_col = role_col
        self.season_col = season_col

    def fit(self, X: pd.DataFrame, y=None):
        self.group_means_: dict[str, pd.Series] = {}
        self.role_means_: dict[str, pd.Series] = {}
        self.global_means_: dict[str, float] = {}

        for col in self.feature_cols:
            if col not in X.columns:
                self.global_means_[col] = 0.0
                continue
            if self.season_col in X.columns and self.role_col in X.columns:
                self.group_means_[col] = X.groupby([self.season_col, self.role_col])[
                    col
                ].mean()
            if self.role_col in X.columns:
                self.role_means_[col] = X.groupby(self.role_col)[col].mean()
            self.global_means_[col] = X[col].mean() if col in X.columns else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col in self.feature_cols:
            if col not in X_out.columns:
                X_out[col] = self.global_means_.get(col, 0.0)
                X_out[f"{col}_missing"] = 1
                continue

            X_out[f"{col}_missing"] = X_out[col].isna().astype(int)
            missing_mask = X_out[col].isna()

            if not missing_mask.any():
                continue

            # Level 1: (season, role) mean
            if (
                col in self.group_means_
                and self.season_col in X_out.columns
                and self.role_col in X_out.columns
            ):
                for idx in X_out[missing_mask].index:
                    key = (
                        X_out.loc[idx, self.season_col],
                        X_out.loc[idx, self.role_col],
                    )
                    if key in self.group_means_[col].index:
                        X_out.loc[idx, col] = self.group_means_[col].loc[key]

            # Level 2: role mean (remaining NaN)
            still_missing = X_out[col].isna()
            if (
                still_missing.any()
                and col in self.role_means_
                and self.role_col in X_out.columns
            ):
                for idx in X_out[still_missing].index:
                    role = X_out.loc[idx, self.role_col]
                    if role in self.role_means_[col].index:
                        X_out.loc[idx, col] = self.role_means_[col].loc[role]

            # Level 3: global mean
            still_missing = X_out[col].isna()
            if still_missing.any():
                X_out.loc[still_missing, col] = self.global_means_.get(col, 0.0)

        return X_out
