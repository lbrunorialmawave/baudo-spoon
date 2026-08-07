"""Optuna hyperparameter tuner with TimeSeriesSplit CV objective.

TimeSeriesSplit is used for ALL CV inside Optuna trials. Random cross-validation
is never used. This is critical: the dataset is chronologically ordered by
season/gameweek, so random splits would leak future data into training.

Nested CV structure:
  - Outer loop: Optuna trials (each trial suggests hyperparams)
  - Inner loop: TimeSeriesSplit(n_splits) CV to evaluate the params
  - Meta-learner trained separately on OOF predictions from best params

This separation ensures the meta-learner's training data (OOF from best
base models) is not contaminated by the hyperparameter search.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

try:
    import optuna as _optuna

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


@dataclass(frozen=True)
class TuningResult:
    """Result of a single Optuna tuning run."""

    model_name: str
    best_params: dict[str, Any]
    best_cv_rmse: float
    n_trials_completed: int
    trial_history: list[dict[str, Any]] = field(default_factory=list)


def _cv_rmse(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> float:
    """Evaluate pipeline via TimeSeriesSplit CV, return mean RMSE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx].values, preds)))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("inf")


class OptunaTuner:
    """Optuna-based hyperparameter search with TimeSeriesSplit CV objective.

    Args:
        n_trials: Number of Optuna trials per model.
        n_splits: Number of TimeSeriesSplit folds for CV objective.
        random_seed: Seed for Optuna sampler.
        timeout: Optional timeout in seconds per study.
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 4,
        random_seed: int = 42,
        timeout: float | None = None,
    ) -> None:
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for OptunaTuner. "
                "Install it with: pip install optuna"
            )
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.timeout = timeout

    def tune(
        self,
        model_name: str,
        pipeline_factory: Callable[[dict[str, Any]], Pipeline],
        param_space: Callable[[Any], dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TuningResult:
        """Run Optuna study for one model.

        Args:
            model_name: Display name for logging.
            pipeline_factory: Callable(params) -> unfitted Pipeline.
            param_space: Callable(trial) -> dict of suggested params.
                Use trial.suggest_float, suggest_int, suggest_categorical.
            X: Training features.
            y: Training target.

        Returns:
            TuningResult with best params and CV RMSE history.
        """
        import optuna

        trial_history: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            params = param_space(trial)
            pipeline = pipeline_factory(params)
            rmse = _cv_rmse(pipeline, X, y, self.n_splits)
            trial_history.append(
                {"trial": trial.number, "params": params, "rmse": rmse}
            )
            return rmse

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best = study.best_trial
        log.info(
            "OptunaTuner [%s]: best RMSE=%.4f after %d trials. Params: %s",
            model_name,
            best.value,
            len(study.trials),
            best.params,
        )

        return TuningResult(
            model_name=model_name,
            best_params=best.params,
            best_cv_rmse=float(best.value),
            n_trials_completed=len(study.trials),
            trial_history=trial_history,
        )

    def tune_ridge(
        self, X: pd.DataFrame, y: pd.Series, preprocessor: Any
    ) -> TuningResult:
        """Convenience: tune Ridge alpha with TimeSeriesSplit CV."""
        from sklearn.linear_model import Ridge as _Ridge

        def param_space(trial: Any) -> dict[str, Any]:
            return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}

        def pipeline_factory(params: dict[str, Any]) -> Pipeline:
            return Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", _Ridge(alpha=params["alpha"])),
                ]
            )

        return self.tune("ridge", pipeline_factory, param_space, X, y)

    def tune_hist_gbm(
        self, X: pd.DataFrame, y: pd.Series, preprocessor: Any
    ) -> TuningResult:
        """Convenience: tune HistGradientBoosting with TimeSeriesSplit CV."""
        from sklearn.ensemble import HistGradientBoostingRegressor as _HGBM

        def param_space(trial: Any) -> dict[str, Any]:
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 63),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            }

        def pipeline_factory(params: dict[str, Any]) -> Pipeline:
            return Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", _HGBM(random_state=self.random_seed, **params)),
                ]
            )

        return self.tune("hist_gbm", pipeline_factory, param_space, X, y)
