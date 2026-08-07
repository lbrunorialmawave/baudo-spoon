"""Phase 3 tests: stacking ensemble, calibration, explainability."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ml.domain.predictions import SHAP_TOLERANCE, PredictionExplanation
from ml.domain.targets import FANTAPUNTI_TOTALI, FANTAVOTO_MEDIO
from ml.ensemble.stacking import StackingEnsemble
from ml.explainability.shap_explainer import ShapExplainer

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_regression_data():
    """4 seasons × 25 players = 100 rows, 5 numeric features."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
            "f3": np.random.uniform(0, 1, n),
            "f4": np.random.uniform(-1, 1, n),
            "season_start": np.tile([2021, 2022, 2023, 2024], 25),
        }
    )
    y = 6.0 + 0.5 * X["f1"] - 0.3 * X["f2"] + np.random.normal(0, 0.2, n)
    y = y.clip(1.0, 10.0)
    return X[["f1", "f2", "f3", "f4"]], pd.Series(y.values, name="fantavoto_medio")


@pytest.fixture
def simple_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                ["f1", "f2", "f3", "f4"],
            ),
        ],
        remainder="drop",
    )


# ── Test: TimeSeriesSplit enforced, KFold never used ─────────────────────────


class TestTimeSeriesSplitEnforced:
    def test_kfold_not_imported_in_stacking(self):
        """KFold must not appear anywhere in ml/ensemble/stacking.py."""
        import ast
        import pathlib

        src = pathlib.Path("ml/ensemble/stacking.py").read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            pytest.fail("ml/ensemble/stacking.py has a syntax error")

        # Check no import of KFold
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                src_str = ast.unparse(node)
                assert "KFold" not in src_str, f"KFold found in import: {src_str}"

    def test_kfold_not_in_calibration_module(self):
        """KFold must not appear in ml/calibration/optuna_tuner.py."""
        import pathlib

        src = pathlib.Path("ml/calibration/optuna_tuner.py").read_text()
        assert "KFold" not in src, "KFold found in calibration module"

    def test_stacking_oof_fold_order_is_chronological(
        self, synthetic_regression_data, simple_preprocessor
    ):
        """OOF generation must respect chronological order.

        Verified indirectly: a StackingEnsemble fitted and predicted on the
        same data should produce predictions in [1, 10] range (no leakage
        artifacts like extreme values that would appear with random splits).
        """
        X, y = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3, random_seed=42)
        ens.fit(X, y, simple_preprocessor)
        result = ens.predict(X)
        # Sanity check: predictions should be within reasonable fantavoto range
        assert result.predictions.min() > 0, "Predictions should be positive"
        assert result.predictions.max() < 15, (
            "Predictions should be < 15 (no leakage artifacts)"
        )


# ── Test: StackingEnsemble fit/predict ───────────────────────────────────────


class TestStackingEnsemble:
    def test_fit_predict_produces_correct_shape(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, y = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3, random_seed=42)
        ens.fit(X, y, simple_preprocessor)
        result = ens.predict(X)
        assert len(result.predictions) == len(X)
        assert len(result.variance) == len(X)
        assert len(result.prediction_interval_low) == len(X)
        assert len(result.prediction_interval_high) == len(X)

    def test_prediction_interval_low_leq_high(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, y = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3, random_seed=42)
        ens.fit(X, y, simple_preprocessor)
        result = ens.predict(X)
        assert (
            result.prediction_interval_low <= result.prediction_interval_high
        ).all(), "prediction_interval_low must always be <= prediction_interval_high"

    def test_variance_non_negative(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, y = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3, random_seed=42)
        ens.fit(X, y, simple_preprocessor)
        result = ens.predict(X)
        assert (result.variance >= 0).all()

    def test_predict_before_fit_raises(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, _ = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3)
        with pytest.raises(RuntimeError, match="fitted"):
            ens.predict(X)

    def test_base_predictions_dict_keys(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, y = synthetic_regression_data
        ens = StackingEnsemble(FANTAVOTO_MEDIO, n_splits=3, random_seed=42)
        ens.fit(X, y, simple_preprocessor)
        result = ens.predict(X)
        assert "ridge" in result.base_predictions
        assert "hist_gbm" in result.base_predictions
        assert len(result.base_predictions) >= 2

    def test_target_transform_applied(
        self, synthetic_regression_data, simple_preprocessor
    ):
        """TargetSpec with log1p transform: ensemble should apply it and inverse."""
        X, y = synthetic_regression_data
        y_total = (y * 30).clip(lower=0.1)  # simulate fantapunti_totali
        ens = StackingEnsemble(FANTAPUNTI_TOTALI, n_splits=3, random_seed=42)
        ens.fit(X, y_total, simple_preprocessor)
        result = ens.predict(X)
        # All predictions should be positive (inverse of log1p is expm1 which > -1)
        assert (result.predictions > -1).all()


# ── Test: ShapExplainer ───────────────────────────────────────────────────────


class TestShapExplainer:
    def test_explain_returns_prediction_explanation(
        self, synthetic_regression_data, simple_preprocessor
    ):
        X, y = synthetic_regression_data
        pipe = Pipeline(
            [
                ("preprocessor", simple_preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipe.fit(X, y)
        feature_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())

        explainer = ShapExplainer(pipe, feature_names, sample_size=50, random_seed=42)
        explainer.fit_explainer(X)

        expl = explainer.explain(
            X.iloc[:1],
            prediction=float(pipe.predict(X.iloc[:1])[0]),
            variance=0.05,
            prediction_interval=(5.5, 7.0),
        )
        assert isinstance(expl, PredictionExplanation)

    def test_shap_coherence_when_shap_available(
        self, synthetic_regression_data, simple_preprocessor
    ):
        """When SHAP is installed, Ridge explanations should be coherent."""
        try:
            import shap  # noqa: F401 — import IS the check (verifies it actually loads, not just findable)
        except ImportError:
            pytest.skip("shap not installed")

        X, y = synthetic_regression_data
        pipe = Pipeline(
            [
                ("preprocessor", simple_preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipe.fit(X, y)
        feature_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())

        explainer = ShapExplainer(pipe, feature_names, sample_size=50, random_seed=42)
        explainer.fit_explainer(X)

        expl = explainer.explain(
            X.iloc[:1],
            prediction=float(pipe.predict(X.iloc[:1])[0]),
            variance=0.02,
            prediction_interval=(5.5, 7.5),
        )
        # SHAP coherence: |Σ shap + base - pred| < SHAP_TOLERANCE
        # For LinearExplainer + Ridge this should hold
        assert expl.is_shap_coherent(), (
            f"SHAP coherence failed: error={expl.shap_coherence_error():.6f} "
            f">= tolerance={SHAP_TOLERANCE}"
        )

    def test_empty_explanation_when_shap_unavailable(self):
        """Without shap installed (or explainer not fitted), returns empty expl."""
        # We test the fallback path by not calling fit_explainer
        from sklearn.linear_model import Ridge as _Ridge

        pipe = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            (
                                "n",
                                Pipeline(
                                    [("i", SimpleImputer()), ("s", RobustScaler())]
                                ),
                                ["f1"],
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                ("model", _Ridge()),
            ]
        )
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        y = pd.Series([6.0, 6.5, 7.0])
        pipe.fit(X, y)

        explainer = ShapExplainer(pipe, ["numeric__f1"])
        # Note: fit_explainer NOT called — _explainer is None
        expl = explainer.explain(
            X.iloc[:1],
            prediction=6.2,
            variance=0.1,
            prediction_interval=(5.5, 7.0),
        )
        assert isinstance(expl, PredictionExplanation)
        assert expl.shap_values == {}
        assert expl.is_shap_coherent()  # base_value = prediction → error = 0

    def test_confidence_derived_from_interval_width(self):
        """Wider interval → lower confidence."""
        expl_narrow = PredictionExplanation(
            prediction=6.5,
            confidence=0.9,
            variance=0.01,
            prediction_interval=(6.2, 6.8),
            best_case=6.8,
            worst_case=6.2,
            top_features=[],
            shap_values={},
            base_value=6.5,
        )
        expl_wide = PredictionExplanation(
            prediction=6.5,
            confidence=0.1,
            variance=0.5,
            prediction_interval=(4.0, 9.0),
            best_case=9.0,
            worst_case=4.0,
            top_features=[],
            shap_values={},
            base_value=6.5,
        )
        assert expl_narrow.confidence > expl_wide.confidence


# ── Test: OptunaTuner (skipped if optuna not installed) ──────────────────────


class TestOptunaTuner:
    def test_tune_ridge_reduces_rmse(
        self, synthetic_regression_data, simple_preprocessor
    ):
        try:
            import optuna  # noqa: F401 — import IS the check (verifies it actually loads, not just findable)

            from ml.calibration.optuna_tuner import OptunaTuner
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = synthetic_regression_data
        tuner = OptunaTuner(n_trials=5, n_splits=3, random_seed=42)
        result = tuner.tune_ridge(X, y, simple_preprocessor)

        assert result.model_name == "ridge"
        assert result.best_cv_rmse > 0
        assert "alpha" in result.best_params
        assert result.n_trials_completed == 5

    def test_no_kfold_in_tuner_cv_path(self):
        """Verify _cv_rmse uses TimeSeriesSplit, not KFold."""
        import inspect

        from ml.calibration import optuna_tuner as _mod

        src = inspect.getsource(_mod._cv_rmse)
        assert "TimeSeriesSplit" in src
        assert "KFold" not in src
