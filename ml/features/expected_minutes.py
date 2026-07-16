"""Expected minutes as a feature for the main ensemble.

ExpectedMinutesFeature wraps a fitted ExpectedMinutesModel and produces
the expected_minutes series for use in the main stacking ensemble.
"""
from __future__ import annotations
import polars as pl
from ml.domain.features import Feature, MissingDataPolicy
from ml.models.expected_minutes import ExpectedMinutesModel


class ExpectedMinutesFeature(Feature):
    """Expected minutes feature backed by a fitted ExpectedMinutesModel.

    Unlike other features, this one requires a pre-fitted model instance.
    The model is trained independently (with its own TimeSeriesSplit backtest)
    before being injected here.

    Args:
        model: Fitted ExpectedMinutesModel instance.
    """

    name = "expected_minutes"
    required_columns = frozenset(["player_fotmob_id", "season_start"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ROLE_MEDIAN

    def __init__(self, model: ExpectedMinutesModel) -> None:
        if model._pipeline is None:
            raise ValueError("ExpectedMinutesFeature requires a fitted ExpectedMinutesModel.")
        self.model = model

    def compute(self, data: pl.DataFrame) -> pl.Series:
        # Convert to pandas for the model, then back to Series
        import pandas as pd
        df_pd = data.to_pandas()
        results = self.model.predict(df_pd)
        values = [r.expected_minutes for r in results]
        return pl.Series("expected_minutes", values, dtype=pl.Float64)
