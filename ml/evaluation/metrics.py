from __future__ import annotations

"""Regression evaluation: metrics, backtesting, and comparison tables.

Metrics computed:
- **RMSE** (Root Mean Squared Error) — penalises large errors more heavily.
- **MAE** (Mean Absolute Error) — interpretable in the target unit (fantavoto).
- **R²** (Coefficient of determination) — fraction of variance explained.

Temporal backtesting:
- ``backtest`` re-trains each model from scratch on progressively expanding
  training windows (one season at a time) and scores on the following season.
  This simulates real-world deployment where only historical data is available
  at prediction time.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SplitMetrics:
    """Metrics for a single train/test or CV fold."""

    rmse: float
    mae: float
    r2: float
    n_test: int

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2, "n_test": float(self.n_test)}


@dataclass
class BacktestResult:
    """Per-season and aggregate backtest results for one model."""

    model_name: str
    season_metrics: list[dict]      # one dict per test season
    mean_rmse: float
    mean_mae: float
    mean_r2: float


# ── Core metric functions ─────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    """Compute RMSE, MAE, R² for a single predictions array."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return SplitMetrics(rmse=rmse, mae=mae, r2=r2, n_test=len(y_true))


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "",
) -> SplitMetrics:
    """Evaluate a fitted pipeline on the held-out test set."""
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)
    log.info(
        "[%s] Test  RMSE=%.4f  MAE=%.4f  R²=%.4f  (n=%d)",
        model_name, metrics.rmse, metrics.mae, metrics.r2, metrics.n_test,
    )
    return metrics


def cv_evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    model_name: str = "",
) -> SplitMetrics:
    """TimeSeriesSplit cross-validation; returns mean metrics across folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        from sklearn.base import clone
        pipe_clone = clone(pipeline)
        pipe_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = pipe_clone.predict(X.iloc[val_idx])
        m = compute_metrics(y.iloc[val_idx].values, y_pred)
        rmses.append(m.rmse)
        maes.append(m.mae)
        r2s.append(m.r2)
        log.debug("  Fold %d: RMSE=%.4f MAE=%.4f R²=%.4f", fold, m.rmse, m.mae, m.r2)

    mean = SplitMetrics(
        rmse=float(np.mean(rmses)),
        mae=float(np.mean(maes)),
        r2=float(np.mean(r2s)),
        n_test=-1,
    )
    log.info(
        "[%s] CV(%d-fold)  RMSE=%.4f  MAE=%.4f  R²=%.4f",
        model_name, n_splits, mean.rmse, mean.mae, mean.r2,
    )
    return mean


# ── Backtesting ───────────────────────────────────────────────────────────────

def backtest(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fantavoto_medio",
    season_col: str = "season_start",
    model_name: str = "",
) -> BacktestResult:
    """Walk-forward backtest: train on seasons t₀…tₙ, test on season tₙ₊₁.

    Args:
        pipeline: An **unfitted** sklearn Pipeline (preprocessor + model).
        df: Full player-season DataFrame including feature_cols and target_col.
        feature_cols: Feature columns used as model inputs.
        target_col: Name of the target column.
        season_col: Name of the season identifier column.
        model_name: Label for logging.

    Returns:
        :class:`BacktestResult` with per-season and aggregate metrics.
    """
    from sklearn.base import clone

    seasons = sorted(df[season_col].unique())
    if len(seasons) < 2:
        log.warning("Backtesting requires ≥2 seasons; skipping.")
        return BacktestResult(
            model_name=model_name,
            season_metrics=[],
            mean_rmse=float("nan"),
            mean_mae=float("nan"),
            mean_r2=float("nan"),
        )

    season_metrics = []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_mask = df[season_col].isin(train_seasons)
        test_mask = df[season_col] == test_season

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col]

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test.values, y_pred)

        log.info(
            "[%s] Backtest season=%d  RMSE=%.4f  MAE=%.4f  R²=%.4f",
            model_name, test_season, m.rmse, m.mae, m.r2,
        )
        season_metrics.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            **m.as_dict(),
        })

    if not season_metrics:
        return BacktestResult(
            model_name=model_name,
            season_metrics=[],
            mean_rmse=float("nan"),
            mean_mae=float("nan"),
            mean_r2=float("nan"),
        )

    return BacktestResult(
        model_name=model_name,
        season_metrics=season_metrics,
        mean_rmse=float(np.mean([s["rmse"] for s in season_metrics])),
        mean_mae=float(np.mean([s["mae"] for s in season_metrics])),
        mean_r2=float(np.mean([s["r2"] for s in season_metrics])),
    )


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(
    results: dict[str, SplitMetrics],
) -> pd.DataFrame:
    """Return a sorted DataFrame comparing all models side-by-side."""
    rows = [
        {"model": name, **m.as_dict()}
        for name, m in results.items()
    ]
    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return df
