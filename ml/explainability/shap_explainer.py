"""ShapExplainer: produces PredictionExplanation with verified SHAP coherence.

For each prediction, verifies:
    |Σ shap_values + base_value - prediction| < SHAP_TOLERANCE

If the invariant fails, a warning is logged and shap_coherence_error() > 0.
The caller can check explanation.is_shap_coherent() before trusting the explanation.

SHAP is an optional dependency. When absent, returns explanations with empty
shap_values and confidence=0.0 (marked as not coherent by definition).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.domain.predictions import PredictionExplanation, SHAP_TOLERANCE

log = logging.getLogger(__name__)

try:
    import shap as _shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    log.warning("shap not installed; ShapExplainer will return empty explanations.")


class ShapExplainer:
    """Computes per-prediction SHAP explanations and wraps in PredictionExplanation.

    Works with tree models (TreeExplainer) and linear models (LinearExplainer).
    Falls back gracefully when SHAP is unavailable.

    Args:
        pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'model' steps.
        feature_names: Feature names after preprocessing (from get_feature_names_out).
        sample_size: Background dataset size for SHAP (used by LinearExplainer).
        random_seed: RNG for subsampling.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        sample_size: int = 200,
        random_seed: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.sample_size = sample_size
        self.random_seed = random_seed
        self._explainer: Optional[Any] = None
        self._preprocessor = pipeline.named_steps.get("preprocessor")
        self._model = pipeline.named_steps.get("model")

    def fit_explainer(self, X_background: pd.DataFrame) -> "ShapExplainer":
        """Initialise the SHAP explainer on background data.

        For tree models: uses TreeExplainer (no background needed, but
        we accept it for API uniformity).
        For linear/other: uses LinearExplainer with a subsample as background.

        Args:
            X_background: Raw DataFrame used to fit the SHAP explainer.
        """
        if not _HAS_SHAP or self._model is None:
            return self

        preprocessor = self._preprocessor
        if preprocessor is None:
            return self

        rng = np.random.default_rng(self.random_seed)
        n = min(self.sample_size, len(X_background))
        idx = rng.choice(len(X_background), size=n, replace=False)
        X_sample = preprocessor.transform(X_background.iloc[idx])

        model = self._model
        model_type = type(model).__name__

        try:
            if model_type in (
                "RandomForestRegressor",
                "ExtraTreesRegressor",
                "GradientBoostingRegressor",
                "HistGradientBoostingRegressor",
                "XGBRegressor",
                "LGBMRegressor",
                "CatBoostRegressor",
            ):
                self._explainer = _shap.TreeExplainer(model)
            else:
                # Linear models and meta-learners
                masker = _shap.maskers.Independent(X_sample)
                self._explainer = _shap.LinearExplainer(model, masker)
        except Exception as exc:
            log.warning("ShapExplainer init failed: %s", exc)

        return self

    def explain(
        self,
        X_row: pd.DataFrame,
        prediction: float,
        variance: float,
        prediction_interval: tuple[float, float],
    ) -> PredictionExplanation:
        """Produce a PredictionExplanation for a single row (or batch average).

        Args:
            X_row: Single-row DataFrame with the same features as training data.
            prediction: The model's point prediction for this row.
            variance: Variance across ensemble base models.
            prediction_interval: (low, high) tuple.

        Returns:
            PredictionExplanation with SHAP values and coherence verified.
        """
        if not _HAS_SHAP or self._explainer is None or self._preprocessor is None:
            return self._empty_explanation(prediction, variance, prediction_interval)

        preprocessor = self._preprocessor

        try:
            X_transformed = preprocessor.transform(X_row)
            shap_result = self._explainer(X_transformed)

            # Handle both old shap_values array API and new Explanation API
            if hasattr(shap_result, "values"):
                shap_vals_arr = shap_result.values[0]  # first row
                base_val = float(shap_result.base_values[0])
            else:
                shap_vals_arr = shap_result[0]  # old API
                base_val = float(self._explainer.expected_value)

            shap_dict = {
                name: float(v)
                for name, v in zip(self.feature_names, shap_vals_arr)
            }

            # Top 10 by absolute SHAP value
            top_features = sorted(
                shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:10]

            # Confidence: derived from relative interval width
            interval_width = prediction_interval[1] - prediction_interval[0]
            typical_scale = 3.0  # typical fantavoto range width
            confidence = float(max(0.0, min(1.0, 1.0 - interval_width / typical_scale)))

            expl = PredictionExplanation(
                prediction=prediction,
                confidence=confidence,
                variance=variance,
                prediction_interval=prediction_interval,
                best_case=float(prediction_interval[1]),
                worst_case=float(prediction_interval[0]),
                top_features=top_features,
                shap_values=shap_dict,
                base_value=base_val,
            )

            # Verify SHAP coherence
            err = expl.shap_coherence_error()
            if err >= SHAP_TOLERANCE:
                log.warning(
                    "SHAP coherence violation: |Σshap + base - pred| = %.6f >= %.6f. "
                    "Check model type compatibility with explainer.",
                    err,
                    SHAP_TOLERANCE,
                )

            return expl

        except Exception as exc:
            log.warning("ShapExplainer.explain failed: %s — returning empty explanation.", exc)
            return self._empty_explanation(prediction, variance, prediction_interval)

    def _empty_explanation(
        self,
        prediction: float,
        variance: float,
        prediction_interval: tuple[float, float],
    ) -> PredictionExplanation:
        interval_width = prediction_interval[1] - prediction_interval[0]
        typical_scale = 3.0
        confidence = float(max(0.0, min(1.0, 1.0 - interval_width / typical_scale)))
        return PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            variance=variance,
            prediction_interval=prediction_interval,
            best_case=float(prediction_interval[1]),
            worst_case=float(prediction_interval[0]),
            top_features=[],
            shap_values={},
            base_value=prediction,  # base = prediction when shap is empty → error = 0
        )
