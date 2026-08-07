"""Multi-source ensemble with versioned weights.

Combines ML model predictions, expert signal, and bookmaker signal via
EnsembleWeightConfig. Logs config version alongside predictions for
backtest traceability.

Usage:
    cfg = EnsembleWeightConfig(version="v1.0", ml_model_weight=0.5,
                                bookmaker_weight=0.3, expert_weight=0.2)
    ensemble = MultiSourceEnsemble(cfg)
    scores = ensemble.combine(ml_scores, expert_scores, bookmaker_scores)

Backtest:
    compare_configs(cfg_a, cfg_b, history_df) shows per-config MAE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ml.ensemble.config import EnsembleWeightConfig

log = logging.getLogger(__name__)

__all__ = ["MultiSourceEnsemble", "compare_configs"]


@dataclass
class MultiSourceEnsemble:
    config: EnsembleWeightConfig

    def combine(
        self,
        ml_scores: pd.Series,
        expert_scores: pd.Series | None = None,
        bookmaker_scores: pd.Series | None = None,
    ) -> pd.Series:
        """Weighted sum of available signals; missing signals get zero weight
        and remaining weights are renormalized.

        Returns a Series with the same index as ml_scores.
        """
        cfg = self.config.normalized()
        result = ml_scores * cfg.ml_model_weight

        active_extra = 0.0
        if expert_scores is not None:
            result = (
                result
                + expert_scores.reindex(ml_scores.index, fill_value=0.0)
                * cfg.expert_weight
            )
            active_extra += cfg.expert_weight
        if bookmaker_scores is not None:
            result = (
                result
                + bookmaker_scores.reindex(ml_scores.index, fill_value=0.0)
                * cfg.bookmaker_weight
            )
            active_extra += cfg.bookmaker_weight

        # renormalize when signals are missing
        total = cfg.ml_model_weight + active_extra
        if total > 0:
            result = result / total

        log.info(
            "MultiSourceEnsemble combine: config_version=%s ml=%.2f expert=%s bookmaker=%s",
            cfg.version,
            cfg.ml_model_weight,
            f"{cfg.expert_weight:.2f}" if expert_scores is not None else "absent",
            f"{cfg.bookmaker_weight:.2f}" if bookmaker_scores is not None else "absent",
        )
        return result.rename(f"ensemble_score__{cfg.version}")


def compare_configs(
    config_a: EnsembleWeightConfig,
    config_b: EnsembleWeightConfig,
    history: pd.DataFrame,
    ml_col: str = "ml_score",
    expert_col: str | None = "expert_rating",
    bookmaker_col: str | None = "bookmaker_signal",
    target_col: str = "actual_score",
) -> pd.DataFrame:
    """Compare two configs on historical data; returns per-config MAE.

    history must have columns: ml_col, target_col, and optionally
    expert_col and bookmaker_col.
    """
    results = []
    for cfg in (config_a, config_b):
        ens = MultiSourceEnsemble(cfg)
        preds = ens.combine(
            ml_scores=history[ml_col],
            expert_scores=history[expert_col]
            if expert_col and expert_col in history.columns
            else None,
            bookmaker_scores=history[bookmaker_col]
            if bookmaker_col and bookmaker_col in history.columns
            else None,
        )
        mae = (preds - history[target_col]).abs().mean()
        results.append({"config_version": cfg.version, "mae": mae})
    return pd.DataFrame(results)
