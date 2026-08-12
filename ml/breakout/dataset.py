"""Breakout dataset construction (PR6 of the low-sample plan).

A **breakout** is defined as a *limited-sample* player (current season
played ``LIMITED`` minutes) who, in the **immediately following season**
``t+1``, plays at least the *standard* threshold of minutes.

Formally::

    breakout(player, season t) = 1
        if cohort(mins(t))   == LIMITED
        and mins(t+1)       >= standard_minutes
        else 0

The label is **strictly forward-looking** and **temporally isolated**:
* No row at season ``t+1`` can influence the features at season ``t``.
* No row at season ``t-1`` can be the *target* of a season-``t`` feature
  row (a player who was LIMITED at ``t-1`` and STANDARD at ``t`` is **not**
  a breakout candidate for the ``t-1`` row — they already broke out).

The dataset is built from the wide player-season DataFrame produced by
:mod:`ml.data.loader`.  It is fully deterministic and side-effect free;
all "side effects" (e.g. writing a Parquet file) are delegated to the
caller.

Key invariants (validated by tests):

* No future-looking feature is used.  Each row contains only the columns
  that already exist at season ``t`` plus the **lagged** version of
  rolling features.
* The label uses ``mins_played`` from the **next** season only.
* A row with no next season (e.g. the most recent scraped season) is
  **excluded** from the training set — keeping it would teach the
  model to predict "no-breakout" for every player that has not yet had
  a chance to break out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from ..sample_reliability.cohort import (
    COHORT_LIMITED,
    classify_cohort,
)

log = logging.getLogger(__name__)


# ── Public constants ────────────────────────────────────────────────────────

DEFAULT_BREAKOUT_TARGET_MINUTES: Final[int] = 800
DEFAULT_FEATURE_LAG_SEASONS: Final[int] = 1


# ── Public DTO ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BreakoutDatasetStats:
    """Summary statistics for a built breakout dataset.

    All fields are JSON-serialisable.  The instance is intended to be
    embedded in the trainer output and in the experiment report.
    """

    n_total: int
    n_positive: int
    n_negative: int
    base_rate: float
    n_excluded_no_next_season: int
    target_minutes: int
    feature_lag_seasons: int


# ── Label construction ──────────────────────────────────────────────────────

def build_breakout_labels(
    df: pd.DataFrame,
    *,
    player_col: str = "player_fotmob_id",
    season_col: str = "season_start",
    minutes_col: str = "mins_played",
    standard_minutes: int = DEFAULT_BREAKOUT_TARGET_MINUTES,
    min_minutes_hard: int = 100,
) -> pd.Series:
    """Return a per-row label series (1 = breakout, 0 = not, NaN = no target).

    Args:
        df: Wide player-season DataFrame.
        player_col: Column identifying each player.
        season_col: Column identifying each season.
        minutes_col: Column with current-season minutes.
        standard_minutes: Threshold for "standard" minutes at ``t+1``.
        min_minutes_hard: Lower eligibility cutoff (LIMITED cohort).

    Returns:
        ``pd.Series[int]`` aligned with ``df.index`` containing:

        * ``1`` if the player is in the LIMITED cohort at season ``t``
          and plays at least ``standard_minutes`` at ``t+1``;
        * ``0`` if the player is in the LIMITED cohort at season ``t``
          and has a recorded ``mins_played`` at ``t+1`` below
          ``standard_minutes``;
        * ``np.nan`` for non-LIMITED rows and for LIMITED rows that
          have no recorded season ``t+1`` (excluded from training).
    """
    if minutes_col not in df.columns:
        raise KeyError(f"Column '{minutes_col}' not found in DataFrame")

    work = df[[player_col, season_col, minutes_col]].copy()
    minutes = pd.to_numeric(work[minutes_col], errors="coerce")
    cohort = minutes.apply(
        lambda m: classify_cohort(
            m,
            min_minutes_hard=min_minutes_hard,
            standard_minutes=standard_minutes,
        )
    )
    work["_cohort"] = cohort

    # Build a per-player mapping of (season → minutes at that season)
    # for O(1) lookups of the "next season" minutes.
    next_season = (
        work[[player_col, season_col, minutes_col]]
        .sort_values([player_col, season_col])
        .copy()
    )
    next_season["_next_minutes"] = pd.to_numeric(
        next_season[minutes_col], errors="coerce"
    )
    lookup = next_season.set_index([player_col, season_col])["_next_minutes"]

    def _label(row: pd.Series) -> float:
        row_cohort = row["_cohort"]
        if row_cohort != COHORT_LIMITED:
            return float("nan")
        next_min = lookup.get((row[player_col], row[season_col] + 1), np.nan)
        if pd.isna(next_min):
            return float("nan")
        return 1.0 if float(next_min) >= standard_minutes else 0.0

    labels = work.apply(_label, axis=1)
    labels.index = df.index
    return labels


# ── Dataset construction ────────────────────────────────────────────────────

def build_breakout_dataset(
    df: pd.DataFrame,
    *,
    player_col: str = "player_fotmob_id",
    season_col: str = "season_start",
    minutes_col: str = "mins_played",
    feature_columns: list[str] | None = None,
    standard_minutes: int = DEFAULT_BREAKOUT_TARGET_MINUTES,
    min_minutes_hard: int = 100,
) -> tuple[pd.DataFrame, pd.Series, BreakoutDatasetStats]:
    """Build a (features, labels, stats) tuple ready for classifier training.

    The returned ``features`` DataFrame only contains columns that
    exist **at season ``t``** — i.e. the input ``df`` is assumed to be
    already engineered with **lagged** rolling features.  Use
    :func:`engineer_breakout_features` to compute those lags on demand.

    Args:
        df: Wide player-season DataFrame.
        feature_columns: Optional whitelist of columns to keep.  When
            ``None``, all columns of ``df`` except identifiers and the
            target minutes are kept.

    Returns:
        Tuple of:
        * ``features`` — DataFrame restricted to ``feature_columns``
          (or all non-identifier columns), restricted to rows where
          the label is not NaN.
        * ``labels`` — aligned ``pd.Series[int]`` (0 or 1).
        * ``stats`` — :class:`BreakoutDatasetStats` summary.
    """
    labels_full = build_breakout_labels(
        df,
        player_col=player_col,
        season_col=season_col,
        minutes_col=minutes_col,
        standard_minutes=standard_minutes,
        min_minutes_hard=min_minutes_hard,
    )
    # Exclude rows without a label (no next season) and non-LIMITED rows.
    valid = labels_full.notna()
    excluded = int((~valid).sum())
    df_labeled = df.loc[valid].copy()
    labels = labels_full.loc[valid].astype(int)

    if feature_columns is None:
        skip = {player_col, season_col, minutes_col}
        feature_columns = [c for c in df.columns if c not in skip]
    features = df_labeled[feature_columns].copy()

    stats = BreakoutDatasetStats(
        n_total=int(len(labels)),
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
        base_rate=float((labels == 1).mean()) if len(labels) else 0.0,
        n_excluded_no_next_season=excluded,
        target_minutes=int(standard_minutes),
        feature_lag_seasons=DEFAULT_FEATURE_LAG_SEASONS,
    )
    log.info(
        "Breakout dataset: %d rows (pos=%d, neg=%d, base_rate=%.3f); "
        "%d rows excluded (no next season / not LIMITED).",
        stats.n_total, stats.n_positive, stats.n_negative,
        stats.base_rate, stats.n_excluded_no_next_season,
    )
    return features, labels, stats


# ── Feature engineering (lagged) ───────────────────────────────────────────

def engineer_breakout_features(
    df: pd.DataFrame,
    *,
    player_col: str = "player_fotmob_id",
    season_col: str = "season_start",
    columns_to_lag: tuple[str, ...] = (
        "mins_played", "starts", "appearances",
    ),
) -> pd.DataFrame:
    """Append lagged versions of *columns_to_lag* to the input frame.

    The output preserves the input row order and adds one new column
    per lagged feature (``<col>_lag1``).  The lag uses the player's own
    previous season, so the frame remains safe to use as a
    classification target with no look-ahead.

    This is intentionally a thin wrapper so that a full PR4 (role
    features) computation can be composed before or after.
    """
    work = df.sort_values([player_col, season_col]).copy()
    for col in columns_to_lag:
        if col not in work.columns:
            continue
        work[f"{col}_lag1"] = work.groupby(player_col)[col].shift(1)
    return work
