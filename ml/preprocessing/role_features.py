"""Role / opportunity features (PR4 of the low-sample plan).

This module adds features that describe **how a player is used** rather
than **how productive** they are.  These are the strongest signals for
predicting whether a low-sample player is about to break out, because:

* a player who is starting regularly but is still producing below their
  expected level is a candidate for positive regression;
* a player whose minutes have been steadily increasing across the last
  few seasons is more likely to "earn" a starter role next season;
* a player whose per-appearance minutes are < 60 typically indicates
  late-game substitutions and should be down-weighted for projection
  purposes.

The transformer enforces the same **temporal isolation** rules as
:mod:`ml.preprocessing.features.RollingFeatureTransformer`:

* Rolling means use ``closed='left'`` so the current season is never
  part of the window.
* Linear-trend slope is computed on the historical (lagged) series.
* Deltas are computed as ``lag1``.

All features are produced on a copy of the input frame; the transformer
is stateless (``fit`` is a no-op) and can be embedded in any sklearn
pipeline.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Default values chosen per plan §12.1: a 3-season window provides
# enough history to smooth out a single injury while remaining
# responsive to the most recent trend signal.
DEFAULT_OPPORTUNITY_WINDOW: Final[int] = 3
DEFAULT_RECENT_WINDOW: Final[int] = 2

# Columns that must be present for the transformer to do anything.
# If a column is missing the corresponding feature is silently skipped
# (the trainer already has per-feature availability guards).
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "mins_played",
    "appearances",
    "starts",
)


class RoleOpportunityFeatureTransformer:
    """Compute role/opportunity features from a player-season DataFrame.

    Args:
        player_col: Column identifying each player across seasons.
        season_col: Column identifying the season (used for sorting).
        opportunity_window: Window (in seasons) for the long-term
            rolling mean of minutes / starts.
        recent_window: Window for the most-recent trend (used for the
            ``*_recent`` features).
    """

    def __init__(
        self,
        player_col: str = "player_fotmob_id",
        season_col: str = "season_start",
        opportunity_window: int = DEFAULT_OPPORTUNITY_WINDOW,
        recent_window: int = DEFAULT_RECENT_WINDOW,
    ) -> None:
        if opportunity_window < 1:
            raise ValueError("opportunity_window must be >= 1")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if recent_window > opportunity_window:
            raise ValueError(
                "recent_window must be <= opportunity_window"
            )
        self.player_col = player_col
        self.season_col = season_col
        self.opportunity_window = opportunity_window
        self.recent_window = recent_window

    # sklearn-compatible API
    def fit(self, X: pd.DataFrame, y=None) -> "RoleOpportunityFeatureTransformer":  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` augmented with opportunity features.

        Produced features (all suffixed with ``_opp`` to be picked up
        downstream by RFE/feature-selection):

        * ``mins_played_opp`` — ``mean`` minutes over the *opportunity*
          window (closed='left' → strictly historical).
        * ``starts_opp`` — ``mean`` starts over the opportunity window.
        * ``appearances_opp`` — ``mean`` appearances over the window.
        * ``mins_played_recent`` — rolling mean over the *recent* window.
        * ``starts_per_appearance_opp`` — starts/appearances, smoothed.
        * ``minutes_per_appearance_opp`` — mins/appearances, smoothed.
        * ``minutes_trend_opp`` — linear slope of minutes over the
          opportunity window (positive = increasing minutes).
        """
        df = X.sort_values([self.player_col, self.season_col]).copy()
        created = 0
        opp = self.opportunity_window
        rec = self.recent_window

        # Long-window rolling means
        for col, suffix in (
            ("mins_played", "mins_played_opp"),
            ("starts", "starts_opp"),
            ("appearances", "appearances_opp"),
        ):
            if col not in df.columns:
                continue
            grp = df.groupby(self.player_col)[col]
            df[suffix] = grp.transform(
                lambda s, w=opp: s.rolling(w, min_periods=1, closed="left").mean()
            )
            created += 1

        # Short-window recent rolling mean
        if "mins_played" in df.columns:
            grp = df.groupby(self.player_col)["mins_played"]
            df["mins_played_recent"] = grp.transform(
                lambda s, w=rec: s.rolling(w, min_periods=1, closed="left").mean()
            )
            created += 1

        # Ratios smoothed by the long window
        for num_col, den_col, suffix in (
            ("starts", "appearances", "starts_per_appearance_opp"),
            ("mins_played", "appearances", "minutes_per_appearance_opp"),
        ):
            if num_col not in df.columns or den_col not in df.columns:
                continue
            ratio = df[num_col] / df[den_col].replace(0, np.nan)
            df[suffix] = (
                ratio.groupby(df[self.player_col])
                .transform(
                    lambda s, w=opp: s.rolling(w, min_periods=1, closed="left").mean()
                )
            )
            created += 1

        # Linear trend of minutes over the opportunity window
        if "mins_played" in df.columns:
            df["minutes_trend_opp"] = (
                df.groupby(self.player_col)["mins_played"]
                .transform(lambda s: _rolling_slope(s, self.opportunity_window))
            )
            created += 1

        log.info(
            "RoleOpportunityFeatureTransformer: %d new features created "
            "(window=%d recent=%d).",
            created, opp, rec,
        )
        return df

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [
            "mins_played_opp",
            "starts_opp",
            "appearances_opp",
            "mins_played_recent",
            "starts_per_appearance_opp",
            "minutes_per_appearance_opp",
            "minutes_trend_opp",
        ]


def add_role_opportunity_features(
    df: pd.DataFrame,
    *,
    player_col: str = "player_fotmob_id",
    season_col: str = "season_start",
    opportunity_window: int = DEFAULT_OPPORTUNITY_WINDOW,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    """Convenience wrapper: returns ``df`` with role/opportunity features."""
    transformer = RoleOpportunityFeatureTransformer(
        player_col=player_col,
        season_col=season_col,
        opportunity_window=opportunity_window,
        recent_window=recent_window,
    )
    return transformer.fit(df).transform(df)


# ── Internal helpers ────────────────────────────────────────────────────────


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Return the OLS slope of ``series`` over a rolling window.

    The slope is computed on a 0..window-1 x-axis (the position within
    the window, not the actual season index) so that a player with
    linearly increasing minutes produces a positive constant slope
    regardless of how the seasons are spaced in time.
    """
    def _slope(values: np.ndarray) -> float:
        if np.all(np.isnan(values)):
            return float("nan")
        # Drop NaN at the start of the window.
        valid = values[~np.isnan(values)]
        if len(valid) < 2:
            return 0.0
        x = np.arange(len(valid), dtype=float)
        y = valid.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom == 0.0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    arr = series.to_numpy(dtype=float, copy=False)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    # Iterate right-to-left so each row sees only historical values.
    for end in range(1, n + 1):
        start = max(0, end - window)
        out[end - 1] = _slope(arr[start:end])
    return pd.Series(out, index=series.index)
