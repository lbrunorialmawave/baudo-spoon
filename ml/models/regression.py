from __future__ import annotations

"""Regression model definitions and training utilities.

Models:
- **LinearRegression** — interpretable baseline; Ridge regularisation for
  stability when features are collinear.
- **RandomForestRegressor** — ensemble of decision trees; resistant to
  over-fitting; built-in feature importance.
- **GradientBoostingRegressor** — sklearn's HistGradientBoosting variant
  (handles NaN natively, much faster on large datasets).
- **XGBRegressor** — XGBoost (if installed); state-of-the-art on tabular data.

Temporal cross-validation:
- ``TimeSeriesSplit`` is used throughout to respect the chronological ordering
  of seasons and prevent data leakage.

Hyperparameter tuning:
- ``RandomizedSearchCV`` with ``TimeSeriesSplit`` is used when ``cfg.tune=True``.
  Optimises negative RMSE.

Design notes:
- All estimators are wrapped in a full sklearn ``Pipeline`` that includes the
  preprocessor, so a single ``.fit(X_df, y)`` call handles everything.
- Returned ``TrainedModel`` objects carry the fitted pipeline, feature names,
  and evaluation metrics so the Trainer can compare models uniformly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from ..config import MLConfig
from .base import ModelSpec

log = logging.getLogger(__name__)

# ── Try to import XGBoost (optional dependency) ───────────────────────────────
try:
    from xgboost import XGBRegressor as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    log.warning("XGBoost not installed; XGBRegressor will be skipped.")


# ── Model registry ─────────────────────────────────────────────────────────────

def _build_registry(random_seed: int) -> dict[str, ModelSpec]:
    registry: dict[str, ModelSpec] = {
        "ridge": ModelSpec(
            name="ridge",
            estimator=Ridge(alpha=1.0),
            param_grid={
                "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
            },
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_seed,
            ),
            param_grid={
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", 0.5],
            },
        ),
        "hist_gbm": ModelSpec(
            name="hist_gbm",
            estimator=HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=random_seed,
            ),
            param_grid={
                "max_iter": [200, 300, 400],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 30],
                # Extended regularisation dims (spec: l2_regularization, max_bins)
                "l2_regularization": [0.0, 0.1, 1.0],
                "max_bins": [128, 255],
            },
        ),
    }

    if _HAS_XGB:
        registry["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=_XGB(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                verbosity=0,
                n_jobs=-1,
            ),
            param_grid={
                "n_estimators": [200, 300, 400],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                # Extended regularisation dims (spec: gamma, min_child_weight, reg_lambda)
                "gamma": [0, 0.1, 0.5, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
        )

    return registry


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrainedModel:
    """A fitted model with its metadata."""

    name: str
    pipeline: Pipeline          # fitted sklearn Pipeline (preprocessor + estimator)
    feature_names: list[str]    # feature names after preprocessing
    metrics: dict[str, float]   # evaluation metrics on the test set
    cv_metrics: dict[str, float] = field(default_factory=dict)  # mean CV metrics


# ── Training ──────────────────────────────────────────────────────────────────

def _make_full_pipeline(
    preprocessor: ColumnTransformer,
    estimator: Any,
) -> Pipeline:
    """Wrap preprocessor + estimator into a single Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _prefix_param_grid(param_grid: dict[str, Any]) -> dict[str, Any]:
    """Prefix param keys with 'model__' for use inside a Pipeline."""
    return {f"model__{k}": v for k, v in param_grid.items()}


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: MLConfig,
) -> dict[str, Pipeline]:
    """Train (and optionally tune) all models in the registry.

    Args:
        X_train: Training features (raw DataFrame, not yet transformed).
        y_train: Target series aligned with X_train.
        preprocessor: An **unfitted** ColumnTransformer.  Each model gets
            its own clone to avoid cross-contamination.
        cfg: Pipeline configuration.

    Returns:
        Dict mapping model name → fitted Pipeline.
    """
    from sklearn.base import clone

    registry = _build_registry(cfg.random_seed)
    tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
    fitted: dict[str, Pipeline] = {}

    for name, spec in registry.items():
        log.info("Training model: %s …", name)
        pipe = _make_full_pipeline(clone(preprocessor), spec.estimator)

        if cfg.tune and spec.param_grid:
            log.info("  Running RandomizedSearchCV (n_iter=%d) …", cfg.tune_iter)
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=_prefix_param_grid(spec.param_grid),
                n_iter=cfg.tune_iter,
                scoring="neg_root_mean_squared_error",
                cv=tscv,
                random_state=cfg.random_seed,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            log.info(
                "  Best params: %s  CV RMSE: %.4f",
                {k.replace("model__", ""): v
                 for k, v in search.best_params_.items()},
                -search.best_score_,
            )
        else:
            pipe.fit(X_train, y_train)

        fitted[name] = pipe
        log.info("  Done.")

    return fitted


# ── Multi-target prediction ────────────────────────────────────────────────────

@dataclass
class MultiTargetResult:
    """Output of :class:`MultiTargetPredictor` prediction.

    Attributes:
        pure_grade_pred: Per-player predicted pure (skill-based) grade.
        bonus_probability_pred: Per-player predicted bonus probability score.
        combined_pred: Weighted combination of both targets.
        grade_weight: Weight applied to ``pure_grade_pred``.
        bonus_weight: Weight applied to ``bonus_probability_pred``.
    """

    pure_grade_pred: np.ndarray
    bonus_probability_pred: np.ndarray
    combined_pred: np.ndarray
    grade_weight: float
    bonus_weight: float


class MultiTargetPredictor:
    """Decouple skill-based performance from high-variance scoring events.

    Trains two independent model chains:
    - **pure_grade chain**: predicts the skill-based component of fantavoto
      (base grade, reflecting a player's overall performance level).
    - **bonus_probability chain**: predicts the bonus-event score (goals,
      assists, and other high-variance fanta-bonus events).

    The final fantavoto projection is a weighted sum::

        predicted_fantavoto = grade_weight * pure_grade
                            + bonus_weight * bonus_probability

    Target derivation when separate columns are absent:

    - ``pure_grade`` ≈ ``fantavoto_medio`` minus scaled attacking contributions
      (goals × 3 + assists × 1 per 90, normalised by appearances).
    - ``bonus_probability`` ≈ the bonus-event complement
      ``fantavoto_medio - pure_grade``.

    Use :meth:`derive_targets` to prepare these columns before calling
    :meth:`fit`.

    Args:
        grade_weight: Weight for the pure-grade component (default 0.65).
        bonus_weight: Weight for the bonus component (default 0.35).
    """

    def __init__(
        self,
        grade_weight: float = 0.65,
        bonus_weight: float = 0.35,
    ) -> None:
        self.grade_weight = grade_weight
        self.bonus_weight = bonus_weight
        self._grade_pipeline: Optional[Pipeline] = None
        self._bonus_pipeline: Optional[Pipeline] = None

    @staticmethod
    def derive_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Derive ``pure_grade`` and ``bonus_probability`` from ``fantavoto_medio``.

        Approximation logic:
        - Bonus events (goals, assists) are scaled back from the fantavoto
          score to isolate the skill (pure grade) component.
        - ``pure_grade`` is clipped to prevent negative values.
        - ``bonus_probability`` is the residual attending to high-variance
          scoring events.

        Args:
            df: Feature-engineered DataFrame containing ``fantavoto_medio``,
                optionally ``goals_per90`` and ``goal_assist_per90``.

        Returns:
            Tuple of ``(pure_grade, bonus_probability)`` pandas Series.
        """
        fv = df["fantavoto_medio"].copy()

        bonus_component = pd.Series(0.0, index=df.index)
        if "goals_per90" in df.columns and "appearances" in df.columns:
            goals_total = df["goals_per90"] * (df["appearances"].clip(lower=1) / 90.0 * 90.0)
            bonus_component += goals_total.fillna(0) * 3.0
        if "goal_assist_per90" in df.columns and "appearances" in df.columns:
            assists_total = df["goal_assist_per90"] * (df["appearances"].clip(lower=1) / 90.0 * 90.0)
            bonus_component += assists_total.fillna(0) * 1.0

        # Normalise bonus component to a per-match scale for comparability
        appearances = df.get("appearances", pd.Series(30.0, index=df.index)).clip(lower=1)
        bonus_per_match = (bonus_component / appearances).clip(lower=0.0, upper=5.0)

        pure_grade = (fv - bonus_per_match).clip(lower=3.0)
        bonus_probability = (fv - pure_grade).clip(lower=0.0)

        return pure_grade, bonus_probability

    def fit(
        self,
        X_train: pd.DataFrame,
        y_pure_grade: pd.Series,
        y_bonus: pd.Series,
        preprocessor: "ColumnTransformer",
        cfg: MLConfig,
    ) -> "MultiTargetPredictor":
        """Train both model chains on the provided targets.

        Args:
            X_train: Training feature DataFrame.
            y_pure_grade: Pure-grade target series.
            y_bonus: Bonus-probability target series.
            preprocessor: Unfitted ``ColumnTransformer``.
            cfg: Pipeline configuration.

        Returns:
            ``self`` (fitted).
        """
        from sklearn.base import clone

        log.info("MultiTargetPredictor: fitting pure_grade chain …")
        grade_pipelines = train_all_models(X_train, y_pure_grade, preprocessor, cfg)
        # Pick best model by lowest MSE on training data (full-data re-fit; no leakage concern)
        self._grade_pipeline = _pick_best_pipeline(grade_pipelines, X_train, y_pure_grade)
        log.info("MultiTargetPredictor: grade chain selected '%s'", self._grade_pipeline_name_)

        log.info("MultiTargetPredictor: fitting bonus_probability chain …")
        bonus_pipelines = train_all_models(X_train, y_bonus, clone(preprocessor), cfg)
        self._bonus_pipeline = _pick_best_pipeline(bonus_pipelines, X_train, y_bonus)
        log.info("MultiTargetPredictor: bonus chain selected '%s'", self._bonus_pipeline_name_)

        return self

    def predict(self, X: pd.DataFrame) -> MultiTargetResult:
        """Generate multi-target predictions and weighted combination.

        Args:
            X: Feature DataFrame aligned with the training feature set.

        Returns:
            :class:`MultiTargetResult` with component and combined predictions.
        """
        if self._grade_pipeline is None or self._bonus_pipeline is None:
            raise RuntimeError("MultiTargetPredictor must be fitted before calling predict()")

        grade_pred = self._grade_pipeline.predict(X)
        bonus_pred = self._bonus_pipeline.predict(X)
        combined = (
            self.grade_weight * grade_pred + self.bonus_weight * bonus_pred
        )
        return MultiTargetResult(
            pure_grade_pred=grade_pred,
            bonus_probability_pred=bonus_pred,
            combined_pred=combined,
            grade_weight=self.grade_weight,
            bonus_weight=self.bonus_weight,
        )

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_steps: int = 8,
    ) -> "MultiTargetPredictor":
        """Find the league-optimal grade/bonus weight ratio via grid search.

        Generates predictions from both fitted chains on a held-out validation
        set, then evaluates RMSE for each candidate ``grade_weight`` in
        ``linspace(0.40, 0.80, n_steps + 1)``.  The pair that minimises
        validation RMSE is stored in ``self.grade_weight`` /
        ``self.bonus_weight``.

        This replaces the static 0.65 / 0.35 split with a data-driven ratio
        tuned to the specific league's scoring distribution.

        Complexity: O(n_steps · N) where N is the length of *X_val*.

        Args:
            X_val: Validation feature DataFrame (must not overlap training data).
            y_val: Validation target series aligned with *X_val*.
            n_steps: Number of grid intervals (grid has ``n_steps + 1`` points).
                A value of 8 gives a 0.05-step grid over [0.40, 0.80].

        Returns:
            ``self`` (in-place update of weights).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._grade_pipeline is None or self._bonus_pipeline is None:
            raise RuntimeError(
                "MultiTargetPredictor must be fitted before calling optimize_weights()"
            )

        grade_pred = self._grade_pipeline.predict(X_val)
        bonus_pred = self._bonus_pipeline.predict(X_val)
        y_true = y_val.values

        best_rmse = float("inf")
        best_gw = self.grade_weight

        for step in range(n_steps + 1):
            gw = 0.40 + step * (0.40 / n_steps)  # [0.40 … 0.80]
            bw = 1.0 - gw
            combined = gw * grade_pred + bw * bonus_pred
            rmse = float(np.sqrt(np.mean((combined - y_true) ** 2)))
            log.debug("optimize_weights: gw=%.2f → val RMSE=%.4f", gw, rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_gw = gw

        self.grade_weight = round(best_gw, 4)
        self.bonus_weight = round(1.0 - best_gw, 4)
        log.info(
            "MultiTargetPredictor.optimize_weights: "
            "grade_weight=%.4f  bonus_weight=%.4f  (val RMSE=%.4f)",
            self.grade_weight, self.bonus_weight, best_rmse,
        )
        return self


def _pick_best_pipeline(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Return the pipeline with the lowest in-sample MSE.

    Used internally by :class:`MultiTargetPredictor` to select which trained
    model chain to keep for each target.

    Args:
        pipelines: Dict of fitted pipelines from :func:`train_all_models`.
        X: Feature DataFrame (training set).
        y: Target series (training set).

    Returns:
        The pipeline with the lowest MSE on *X, y*.
    """
    best_mse = float("inf")
    best_pipe: Optional[Pipeline] = None
    best_pipe_name_ = ""

    for name, pipe in pipelines.items():
        preds = pipe.predict(X)
        mse = float(np.mean((preds - y.values) ** 2))
        log.debug("_pick_best_pipeline: %s MSE=%.4f", name, mse)
        if mse < best_mse:
            best_mse = mse
            best_pipe = pipe
            best_pipe_name_ = name

    assert best_pipe is not None
    # Stash name for logging without adding a dataclass field
    best_pipe._multi_target_name = best_pipe_name_  # type: ignore[attr-defined]

    # Assign back so fit() can log it via the hack above
    # (avoids needing a separate return value)
    return best_pipe


# Monkey-patch to expose _grade/bonus_pipeline_name_ on MultiTargetPredictor
def _mp_grade_name(self) -> str:
    return getattr(self._grade_pipeline, "_multi_target_name", "unknown")

def _mp_bonus_name(self) -> str:
    return getattr(self._bonus_pipeline, "_multi_target_name", "unknown")

MultiTargetPredictor._grade_pipeline_name_ = property(_mp_grade_name)  # type: ignore[attr-defined]
MultiTargetPredictor._bonus_pipeline_name_ = property(_mp_bonus_name)  # type: ignore[attr-defined]
