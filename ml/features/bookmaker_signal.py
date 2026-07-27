"""Bookmaker signal feature.

Proxy for probability-of-starting / match impact derived from Snai 1X2 and
Over-Under odds already scraped into the DB.

Extends team_strength.py data rather than duplicating it: team-level odds
are converted to a per-player participation probability proxy.

Expected input columns (pre-joined by loader):
  - ``team_win_prob``   : P(team wins) from 1X2 home/away odds
  - ``over_2_5_prob``   : P(over 2.5 goals) as market liquidity proxy

# ponytail: equal weight for win_prob and over_2_5 for now; calibrate per
# season once we have enough historical odds data to validate.
"""
from __future__ import annotations

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = ["BookmakerSignalFeature"]


class BookmakerSignalFeature(Feature):
    """Normalised bookmaker signal in [0.0, 1.0].

    Combines team win probability and over-2.5 market signal (equal weight).
    Missing values are imputed with the role median.
    """

    name = "bookmaker_signal"
    required_columns = frozenset(["team_win_prob", "over_2_5_prob"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ROLE_MEDIAN

    def compute(self, data: pl.DataFrame) -> pl.Series:
        win = data["team_win_prob"].cast(pl.Float64).clip(0.0, 1.0)
        over = data["over_2_5_prob"].cast(pl.Float64).clip(0.0, 1.0)
        combined = (win + over) / 2.0
        c_min = combined.min() or 0.0
        c_max = combined.max() or 1.0
        if c_max == c_min:
            return pl.Series(self.name, [0.5] * len(data), dtype=pl.Float64)
        return (combined - c_min) / (c_max - c_min)
