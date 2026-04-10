from __future__ import annotations

"""Feature engineering and feature selection for the fantasy-football pipeline.

This module transforms the wide-format player-season DataFrame produced by
the data layer into a richer feature set suitable for regression and
clustering.

Feature groups produced:
1. **Per-90 stats** — all numeric FotMob stats normalised by minutes played
   (or divided by appearances when minutes are unavailable).
2. **Career trend: rolling averages** — 2-season rolling mean of selected
   per-90 stats (requires ≥2 seasons of data per player).  Uses
   ``closed='left'`` to enforce strict temporal isolation (no look-ahead bias).
3. **Career trend: year-over-year delta** — change in key stats vs. prior season.
4. **Schedule-Adjusted Performance (SAP)** — per-90 stats weighted by mean
   opponent ``team_rank_norm`` to normalise for schedule difficulty.
5. **Team context** — already present after the loader merge; kept as-is.
6. **Season index** — ordinal encoding of season (0 = earliest scraped).

Assumptions:
- Input DataFrame has columns from `data.loader.load_raw_data` (wide format).
- FotMob category slugs use camelCase (e.g. "minutesPlayed", "goalAssist").
- Rows are deduplicated: one row per (player_fotmob_id, season_start).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

# ── Data-quality threshold ─────────────────────────────────────────────────────
# Player-seasons with fewer minutes than this are excluded to mitigate per-90
# noise and statistical outliers from low-participation entries.
_MIN_MINUTES_THRESHOLD: int = 450

# ── Constants ─────────────────────────────────────────────────────────────────

# Stats that get per-90 normalisation.  Only columns present in the
# DataFrame will be processed; missing ones are silently skipped.
# All names must match canonical output of stat_names.canonicalize_columns().
_PER_90_CANDIDATES = [
    # Attacking
    "goals", "goal_assist",
    "total_scoring_att", "ontarget_scoring_att",
    "big_chance_created", "big_chance_missed",
    "total_att_assist",
    "won_contest",
    # Set pieces / disciplinary
    "yellow_card", "red_card",
    "penalty_won", "penalty_conceded",
    # Defensive
    "outfielder_block", "interception",
    "total_tackle", "effective_clearance",
    "accurate_pass",
    "fouls",
    # Goalkeeper
    "saves", "_goals_prevented", "goals_conceded", "clean_sheet",
]

# Stats used for rolling/delta trend features.
_TREND_CANDIDATES = [
    "goals_per90", "goal_assist_per90",
    "total_scoring_att_per90", "ontarget_scoring_att_per90",
    "yellow_card_per90",
    "won_contest_per90",
    "total_att_assist_per90",
    "interception_per90",
    "saves_per90",
    "_goals_prevented_per90",
]

# Stats to apply schedule-adjusted performance (SAP) weighting to.
_SAP_STAT_COLS: list[str] = [
    "goals_per90",
    "goal_assist_per90",
    "total_scoring_att_per90",
    "ontarget_scoring_att_per90",
    "saves_per90",
    "_goals_prevented_per90",
]

# Role ordinal encoding: consistent across training and inference.
_ROLE_ORDINAL: dict[str, int] = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# Environmental (contextual) stats — filled with per-column median rather than 0,
# because a missing value means "unknown context", not "zero context".
# Event-based stats (goals, assists, cards, …) retain fillna(0) in add_per90_features.
_ENVIRONMENTAL_STAT_COLS: list[str] = [
    "team_strength_score",
    "is_top_team",
    "team_rank_norm",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_minutes(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in ("mins_played", "minutesPlayed"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").clip(lower=1)
    return None


def _get_appearances(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in ("appearances", "matchesPlayed"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").clip(lower=1)
    return None


def _denominator_per90(df: pd.DataFrame) -> pd.Series:
    """Return per-90-minute denominator (minutes / 90, or appearances)."""
    mins = _get_minutes(df)
    if mins is not None:
        return (mins / 90.0).clip(lower=1)
    apps = _get_appearances(df)
    if apps is not None:
        log.warning(
            "minutesPlayed not found; using appearances as per-90 denominator."
        )
        return apps
    raise ValueError(
        "Neither 'mins_played' nor 'appearances' found. "
        "Cannot compute per-90 features."
    )


# ── Data quality gate ─────────────────────────────────────────────────────────

def filter_min_minutes(
    df: pd.DataFrame,
    min_minutes: int = _MIN_MINUTES_THRESHOLD,
) -> pd.DataFrame:
    """Exclude player-seasons with fewer than *min_minutes* minutes played.

    Removes low-participation entries whose per-90 statistics are unreliable
    due to small sample sizes.

    Args:
        df: Raw player-season DataFrame containing a minutes column.
        min_minutes: Minimum minutes threshold (default 450).

    Returns:
        A new DataFrame with under-threshold rows dropped.
    """
    minutes = _get_minutes(df)
    if minutes is None:
        log.warning(
            "filter_min_minutes: no minutes column found; skipping filter."
        )
        return df.copy()
    mask = minutes >= min_minutes
    n_dropped = int((~mask).sum())
    if n_dropped:
        log.info(
            "filter_min_minutes: dropped %d rows with < %d minutes played.",
            n_dropped, min_minutes,
        )
    return df[mask].copy()


# ── Differentiated imputation ──────────────────────────────────────────────────

def impute_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill environmental (contextual) stat NaNs with their per-column median.

    Event-based stats (goals, assists, cards, …) are already filled with 0
    inside :func:`add_per90_features`.  Environmental stats representing team
    context should use the population median to avoid phantom zero-context bias.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        A new copy of *df* with :data:`_ENVIRONMENTAL_STAT_COLS` imputed.
    """
    df = df.copy()
    for col in _ENVIRONMENTAL_STAT_COLS:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.debug("impute_environmental_features: filled '%s' with median %.4f", col, med)
    return df


# ── Per-90 normalisation ──────────────────────────────────────────────────────

def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append *_per90 columns for each stat in ``_PER_90_CANDIDATES``.

    Returns a new copy of *df*.
    """
    df = df.copy()
    denom = _denominator_per90(df)
    created = []
    for col in _PER_90_CANDIDATES:
        if col in df.columns:
            new_col = f"{col}_per90"
            df[new_col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / denom
            created.append(new_col)
    log.info("Created %d per-90 features.", len(created))
    return df


# ── Rolling / trend features ──────────────────────────────────────────────────

class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for rolling temporal features.

    Uses ``closed='left'`` in the rolling window, which means each row's
    window spans ``[current_pos - window, current_pos)`` — strictly
    historical, never including the current or future season.  This
    eliminates look-ahead bias and makes the transformer safe inside
    sklearn cross-validation pipelines.

    The transformer is stateless: ``fit()`` is a no-op.

    Args:
        window: Size of the historical rolling window (in seasons).
        player_col: Column identifying the player across seasons.
        season_col: Column identifying the season (used for sorting only).
    """

    def __init__(
        self,
        window: int = 2,
        player_col: str = "player_fotmob_id",
        season_col: str = "season_start",
    ) -> None:
        self.window = window
        self.player_col = player_col
        self.season_col = season_col

    def fit(self, X: pd.DataFrame, y=None) -> "RollingFeatureTransformer":  # noqa: ARG002
        """No-op: this transformer has no fit-time state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling-mean and delta features with temporal isolation.

        ``closed='left'`` guarantees that the rolling window never includes
        the current season's value or any future-season values.
        """
        df = X.sort_values([self.player_col, self.season_col]).copy()
        available = [c for c in _TREND_CANDIDATES if c in df.columns]

        if not available:
            log.warning(
                "No trend feature candidates found. Run add_per90_features first."
            )
            return df

        window = self.window
        rolling_created = 0
        delta_created = 0

        for col in available:
            grp = df.groupby(self.player_col)[col]

            # Rolling mean: closed='left' → window [i-window, i) → strictly historical
            roll_col = f"{col}_roll{window}"
            df[roll_col] = grp.transform(
                lambda s, w=window: s.rolling(w, min_periods=1, closed="left").mean()
            )
            rolling_created += 1

            # Year-over-year delta: diff(1) uses previous season only
            delta_col = f"{col}_delta1"
            df[delta_col] = grp.transform(lambda s: s.diff(1))
            delta_created += 1

        log.info(
            "RollingFeatureTransformer: %d rolling + %d delta features.",
            rolling_created, delta_created,
        )
        return df


def add_trend_features(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Append rolling-mean and year-over-year delta features.

    Delegates to :class:`RollingFeatureTransformer` to guarantee temporal
    isolation via ``closed='left'``.  Players with insufficient history
    will have NaN rolling values, handled by downstream imputation.

    Returns a new copy of *df*.
    """
    transformer = RollingFeatureTransformer(window=window)
    return transformer.fit(df).transform(df)


# ── Schedule-Adjusted Performance ─────────────────────────────────────────────

class OpponentStrengthAdjuster(BaseEstimator, TransformerMixin):
    """Weight per-90 stats by the mean opponent schedule strength.

    For each player in a (season, league) group, their **schedule
    difficulty** is approximated as the mean ``team_rank_norm`` of all
    OTHER teams in the same league-season:

    .. code-block:: text

        opponent_mean_rank = (league_total_rank - own_rank) / (n_teams - 1)
        sap_weight = opponent_mean_rank / league_mean_rank

    A ``sap_weight`` > 1 means the player faced stronger-than-average
    opposition.  The adjusted stats (suffixed ``_sap``) are rescaled
    accordingly.

    Complexity: O(N) via vectorised ``groupby`` operations.

    Args:
        group_cols: Columns identifying a league-season cohort.
            Defaults to ``["season_start", "league_name"]``.
    """

    def __init__(self, group_cols: list[str] | None = None) -> None:
        self.group_cols = group_cols or ["season_start", "league_name"]

    def fit(self, X: pd.DataFrame, y=None) -> "OpponentStrengthAdjuster":  # noqa: ARG002
        """Compute per-(league, season) rank aggregates from training data."""
        group_cols = self.group_cols
        if "team_rank_norm" not in X.columns or not all(
            c in X.columns for c in group_cols
        ):
            log.warning(
                "OpponentStrengthAdjuster: required columns missing; "
                "SAP features will be skipped."
            )
            self.league_stats_: pd.DataFrame | None = None
            return self

        # Deduplicate to one row per (team, season) before aggregation so that
        # teams with many players don't inflate the league totals.
        dedup = (
            X[group_cols + ["team_fotmob_id", "team_rank_norm"]]
            .drop_duplicates(subset=group_cols + ["team_fotmob_id"])
        )
        self.league_stats_ = (
            dedup.groupby(group_cols)["team_rank_norm"]
            .agg(league_sum="sum", n_teams="count", league_mean="mean")
            .reset_index()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply SAP weights; returns a new copy of *X* with ``*_sap`` columns."""
        if not hasattr(self, "league_stats_") or self.league_stats_ is None:
            return X.copy()

        group_cols = self.group_cols
        if not all(c in X.columns for c in group_cols):
            return X.copy()

        out = X.copy()
        out = out.merge(self.league_stats_, on=group_cols, how="left")

        # Fallback for unseen (season, league) combos in the test set
        global_mean = float(self.league_stats_["league_mean"].mean())
        out["league_mean"] = out["league_mean"].fillna(global_mean)
        out["league_sum"] = out["league_sum"].fillna(global_mean)
        out["n_teams"] = out["n_teams"].fillna(10.0)

        # Opponent mean rank = (league total − own rank) / (n_teams − 1)
        n_minus_1 = (out["n_teams"] - 1.0).clip(lower=1.0)
        own_rank = out["team_rank_norm"].fillna(global_mean)
        opp_mean = (out["league_sum"] - own_rank) / n_minus_1

        # Normalise so that average-schedule → weight 1.0
        out["sap_weight"] = (
            (opp_mean / out["league_mean"].clip(lower=1e-6))
            .clip(lower=0.1, upper=10.0)
        )

        sap_cols = [c for c in _SAP_STAT_COLS if c in out.columns]
        for col in sap_cols:
            out[f"{col}_sap"] = out[col] * out["sap_weight"]

        out = out.drop(
            columns=["league_sum", "n_teams", "league_mean", "sap_weight"],
            errors="ignore",
        )
        log.info(
            "OpponentStrengthAdjuster: created %d SAP features.", len(sap_cols)
        )
        return out


def add_opponent_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :class:`OpponentStrengthAdjuster` fit-transform in-place.

    Returns a new copy of *df* with ``*_sap`` columns appended.
    Gracefully skips if required columns are absent.
    """
    try:
        adjuster = OpponentStrengthAdjuster()
        return adjuster.fit(df).transform(df)
    except Exception as exc:  # noqa: BLE001
        log.warning("SAP feature engineering failed: %s — skipping.", exc)
        return df.copy()


# ── Outlier handling ──────────────────────────────────────────────────────────

def cap_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Winsorise numeric columns at the given quantile.

    Returns a new copy of *df*.
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            upper = df[col].quantile(quantile)
            df[col] = df[col].clip(upper=upper)
    return df


# ── Role encoding ─────────────────────────────────────────────────────────────

def add_role_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``role_code`` ordinal column from ``canonical_role``.

    GK=0, DEF=1, MID=2, FWD=3.  Rows with missing/unknown role map to 3 (FWD).
    Returns a new copy of *df*.
    """
    df = df.copy()
    if "canonical_role" not in df.columns:
        log.warning(
            "canonical_role column not found; role_code will default to 3 (FWD)."
        )
        df["role_code"] = 3
        return df

    df["role_code"] = (
        df["canonical_role"]
        .map(_ROLE_ORDINAL)
        .fillna(3)
        .astype(int)
    )
    return df


# ── Season index ──────────────────────────────────────────────────────────────

def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``season_idx`` (0-based ordinal of season_start, ascending).

    Returns a new copy of *df*.
    """
    df = df.copy()
    seasons = sorted(df["season_start"].unique())
    mapping = {s: i for i, s in enumerate(seasons)}
    df["season_idx"] = df["season_start"].map(mapping)
    return df


# ── Master transform ──────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    trend_window: int = 2,
    min_minutes: int = _MIN_MINUTES_THRESHOLD,
) -> pd.DataFrame:
    """Run all feature engineering steps in order.

    Steps:
    0. Small-sample filter — drop player-seasons with < *min_minutes* played.
    1. Per-90 normalisation (event-based NaNs → 0).
    2. Temporal rolling features (``closed='left'`` — no look-ahead).
    3. Schedule-adjusted performance (SAP) weighting.
    4. Season ordinal index.
    5. Role ordinal encoding.
    6. Environmental stat imputation (contextual NaNs → median).
    7. Winsorise per-90 stats at the 99th percentile.

    Args:
        df: Raw player-season DataFrame.
        trend_window: Rolling window size in seasons.
        min_minutes: Minimum minutes required to retain a row (default 450).

    Returns:
        A new copy of *df* with all engineered columns appended.
    """
    df = df.copy()
    df = filter_min_minutes(df, min_minutes=min_minutes)
    df = add_per90_features(df)
    df = add_trend_features(df, window=trend_window)
    df = add_opponent_strength_features(df)
    df = add_season_index(df)
    df = add_role_encoding(df)
    df = impute_environmental_features(df)

    # Winsorise per-90 stats at 99th percentile
    per90_cols = [c for c in df.columns if c.endswith("_per90")]
    df = cap_outliers(df, per90_cols, quantile=0.99)

    return df


# ── Feature column selection ──────────────────────────────────────────────────

# Non-feature columns that must be excluded from the model input matrix.
_META_COLS = {
    "player_fotmob_id", "player_name", "team_fotmob_id", "team_name",
    "season_start", "season_label", "league_name", "season_idx",
    "fantavoto_medio",
}

# Categorical columns to one-hot encode.
CATEGORICAL_FEATURES: list[str] = ["league_name"]

# Numeric features selected in priority order. Columns absent from the
# DataFrame are silently dropped by the pipeline's ColumnTransformer.
# All names must match the output of add_per90_features() + add_role_encoding().
NUMERIC_FEATURE_CANDIDATES: list[str] = [
    # ── Role ─────────────────────────────────────────────────────────────────
    "role_code",
    # ── Per-90 attacking ─────────────────────────────────────────────────────
    "goals_per90", "goal_assist_per90",
    "total_scoring_att_per90", "ontarget_scoring_att_per90",
    "big_chance_created_per90", "big_chance_missed_per90",
    "total_att_assist_per90",
    "won_contest_per90",
    # ── Per-90 disciplinary ───────────────────────────────────────────────────
    "yellow_card_per90", "red_card_per90",
    "penalty_scored_per90", "penalty_missed_per90",
    "own_goals_per90",
    # ── Per-90 defensive ─────────────────────────────────────────────────────
    "total_tackle_per90", "interception_per90",
    "effective_clearance_per90", "outfielder_block_per90",
    # ── Per-90 goalkeeper ────────────────────────────────────────────────────
    "saves_per90", "_goals_prevented_per90",
    "goals_conceded_per90", "clean_sheet_per90",
    # ── Schedule-adjusted performance (SAP) ──────────────────────────────────
    "goals_per90_sap", "goal_assist_per90_sap",
    "total_scoring_att_per90_sap", "ontarget_scoring_att_per90_sap",
    "saves_per90_sap", "_goals_prevented_per90_sap",
    # ── Team context ──────────────────────────────────────────────────────────
    "team_strength_score", "is_top_team", "team_rank_norm",
    # ── Season / temporal ─────────────────────────────────────────────────────
    "season_idx",
    # ── Raw counts ────────────────────────────────────────────────────────────
    "appearances", "mins_played",
    # ── Trend features (NaN when only 1 season available) ────────────────────
    "goals_per90_roll2", "goal_assist_per90_roll2",
    "total_scoring_att_per90_roll2", "saves_per90_roll2",
    "_goals_prevented_per90_roll2",
    "goals_per90_delta1", "goal_assist_per90_delta1",
    "yellow_card_per90_delta1",
]


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) that exist in *df*.

    Only returns columns that are present in the DataFrame and contain
    at least one non-NaN value.
    """
    numeric = [
        c for c in NUMERIC_FEATURE_CANDIDATES
        if c in df.columns and df[c].notna().any()
    ]
    categorical = [
        c for c in CATEGORICAL_FEATURES
        if c in df.columns and df[c].notna().any()
    ]
    log.info(
        "Feature selection: %d numeric + %d categorical features.",
        len(numeric), len(categorical),
    )
    return numeric, categorical


# ── Recursive Feature Elimination ─────────────────────────────────────────────

def select_features_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    n_features_fraction: float = 0.70,
) -> list[str]:
    """Prune collinear numeric features via Recursive Feature Elimination.

    Uses ``Ridge`` regression (via ``RobustScaler`` pre-scaling) as the base
    estimator.  RFE iteratively removes the feature with the smallest absolute
    coefficient until the target count is reached.

    Only applies when there are at least 4 candidate features; returns the
    original list unchanged otherwise (avoids unnecessary computation on small
    datasets).

    Complexity: O(k · N · F) where k is the number of elimination rounds,
    N is training rows, and F is the number of features.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target series aligned with *X_train*.
        numeric_features: Candidate numeric feature names to evaluate.
        n_features_fraction: Fraction of features to retain (default 0.70).
            The actual count is ``max(1, round(len(available) * fraction))``.

    Returns:
        Ordered list of selected numeric feature names.  Always a non-empty
        subset of *numeric_features*.
    """
    available = [
        f for f in numeric_features
        if f in X_train.columns and X_train[f].notna().any()
    ]
    if len(available) < 4:
        log.debug("select_features_rfe: too few features (%d); skipping RFE.", len(available))
        return available

    n_to_keep = max(1, int(round(len(available) * n_features_fraction)))
    if n_to_keep >= len(available):
        return available

    X_num = X_train[available].copy()
    col_medians = X_num.median()
    X_num = X_num.fillna(col_medians)

    # IQR-based scaling before RFE so that Ridge coefficients are comparable
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_num)

    rfe = RFE(
        estimator=Ridge(alpha=1.0),
        n_features_to_select=n_to_keep,
        step=1,
    )
    rfe.fit(X_scaled, y_train.values)

    selected = [f for f, support in zip(available, rfe.support_) if support]
    log.info(
        "RFE: %d → %d numeric features retained (fraction=%.2f).",
        len(available), len(selected), n_features_fraction,
    )
    return selected
