"""Team context features.

These features read from pre-computed team-strength columns produced by
``ml/data/loader.py``.  They are stateless lookups — no computation beyond
normalisation.
"""

from __future__ import annotations

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = ["IsTopTeamFeature", "TeamRankNormFeature", "TeamStrengthFeature"]


class TeamStrengthFeature(Feature):
    """Normalised team strength score in [0.0, 1.0].

    Min-max normalises ``team_strength_score`` within the dataset.
    Falls back to 0.5 (neutral) when all values are identical.
    """

    name = "team_strength"
    required_columns = frozenset(["team_strength_score"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ROLE_MEDIAN

    def compute(self, data: pl.DataFrame) -> pl.Series:
        s = data["team_strength_score"].cast(pl.Float64)
        s_min = s.min() or 0.0
        s_max = s.max() or 1.0
        if s_max == s_min:
            return pl.Series("team_strength", [0.5] * len(data), dtype=pl.Float64)
        return (s - s_min) / (s_max - s_min)


class IsTopTeamFeature(Feature):
    """Binary flag: 1.0 if player is in a top-3 team by season, else 0.0."""

    name = "is_top_team"
    required_columns = frozenset(["is_top_team"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    def compute(self, data: pl.DataFrame) -> pl.Series:
        return data["is_top_team"].cast(pl.Float64).fill_null(0.0)


class TeamRankNormFeature(Feature):
    """Normalised team rank within season-league (0 = worst, 1 = best)."""

    name = "team_rank_norm"
    required_columns = frozenset(["team_rank_norm"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ROLE_MEDIAN

    def compute(self, data: pl.DataFrame) -> pl.Series:
        return data["team_rank_norm"].cast(pl.Float64).fill_null(0.5)
