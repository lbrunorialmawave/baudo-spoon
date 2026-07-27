"""Expert rating signal feature.

Computes per-player weighted average of expert ratings from the
``expert_ratings`` table (multiple sources). Keeps expert signal as a
separate feature column, not pre-aggregated with ML output.

Expected input columns: ``expert_rating_weighted_avg`` (pre-joined by loader)
or ``expert_rating`` with equal-weight fallback.

# ponytail: equal source weights for now; derive from EnsembleWeightConfig
# when per-source reliability data is available.
"""
from __future__ import annotations

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = ["ExpertRatingFeature"]


class ExpertRatingFeature(Feature):
    """Normalised expert rating in [0.0, 1.0].

    Reads ``expert_rating_weighted_avg`` if present, falls back to
    ``expert_rating``.  Missing values are imputed with the role median.
    """

    name = "expert_rating"
    required_columns = frozenset(["expert_rating_weighted_avg"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ROLE_MEDIAN

    def compute(self, data: pl.DataFrame) -> pl.Series:
        col = (
            "expert_rating_weighted_avg"
            if "expert_rating_weighted_avg" in data.columns
            else "expert_rating"
        )
        s = data[col].cast(pl.Float64)
        s_min = s.min() or 0.0
        s_max = s.max() or 1.0
        if s_max == s_min:
            return pl.Series(self.name, [0.5] * len(data), dtype=pl.Float64)
        return (s - s_min) / (s_max - s_min)
