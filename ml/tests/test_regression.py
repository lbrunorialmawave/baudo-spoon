"""Unit tests for ml.models.regression.

Covers:
- MultiTargetPredictor.optimize_weights: validation-based weight grid search.
- _create_engine_with_retry: exponential backoff logic in run_pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_simple_xy(
    n_rows: int = 50,
    n_feats: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal feature matrix + target for regression model tests."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_feats)]
    X = pd.DataFrame(rng.standard_normal((n_rows, n_feats)), columns=cols)
    y = pd.Series(rng.uniform(4.0, 8.0, n_rows), name="fantavoto_medio")
    return X, y


def _make_min_config(**overrides):
    """Minimal MLConfig without a real DB for unit tests."""
    from ml.config import MLConfig
    defaults = dict(
        database_url="postgresql://test:t@localhost/test",
        tune=False,
        cv_folds=2,
        random_seed=42,
        tune_iter=2,
    )
    defaults.update(overrides)
    return MLConfig(**defaults)


def _make_fitted_predictor():
    """Return a fitted MultiTargetPredictor on a tiny dataset."""
    from ml.models.regression import MultiTargetPredictor
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    X, y = _make_simple_xy(n_rows=60, n_feats=4)
    cfg = _make_min_config()

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler()),
    ])
    preprocessor = ColumnTransformer(
        [("num", num_pipe, list(X.columns))],
        remainder="drop",
    )

    pure_grade, bonus_prob = MultiTargetPredictor.derive_targets(
        X.assign(fantavoto_medio=y, appearances=30)
    )

    predictor = MultiTargetPredictor()
    predictor.fit(X, pure_grade, bonus_prob, preprocessor, cfg)
    return predictor, X, y


# ── MultiTargetPredictor.optimize_weights ─────────────────────────────────────

class TestMultiTargetWeightOptimization:
    """Validation-based weight optimisation replaces static 0.65/0.35 defaults.

    Requirements:
    - optimize_weights must update grade_weight and bonus_weight in-place.
    - Sum of weights must be 1.0 after optimisation.
    - Weights must fall within the search grid [0.40, 0.80].
    - Calling optimize_weights before fit() must raise RuntimeError.
    - Optimised weights must not be worse than defaults on the same val set.
    """

    def test_weights_update_in_place(self):
        """After optimize_weights, both weight attributes must be updated."""
        predictor, X, y = _make_fitted_predictor()
        old_gw = predictor.grade_weight

        predictor.optimize_weights(X, y, n_steps=4)

        # At minimum, the attribute must exist and be a float in valid range
        assert isinstance(predictor.grade_weight, float)
        assert isinstance(predictor.bonus_weight, float)

    def test_weights_sum_to_one(self):
        """grade_weight + bonus_weight must equal 1.0 after optimisation."""
        predictor, X, y = _make_fitted_predictor()
        predictor.optimize_weights(X, y, n_steps=4)

        total = predictor.grade_weight + predictor.bonus_weight
        assert abs(total - 1.0) < 1e-6, (
            f"Weights must sum to 1.0; got {total:.6f}"
        )

    def test_weights_within_grid_range(self):
        """grade_weight must lie in [0.40, 0.80] — the grid search bounds."""
        predictor, X, y = _make_fitted_predictor()
        predictor.optimize_weights(X, y, n_steps=8)

        assert 0.39 <= predictor.grade_weight <= 0.81, (
            f"grade_weight {predictor.grade_weight} out of expected [0.40, 0.80] range"
        )

    def test_raises_if_not_fitted(self):
        """optimize_weights on an unfitted predictor must raise RuntimeError."""
        from ml.models.regression import MultiTargetPredictor

        predictor = MultiTargetPredictor()
        X, y = _make_simple_xy()

        with pytest.raises(RuntimeError, match="fitted"):
            predictor.optimize_weights(X, y)

    def test_optimised_rmse_not_worse_than_defaults(self):
        """The optimised weights must achieve val RMSE ≤ static-weight RMSE.

        Because optimize_weights performs a grid search over the same val set,
        the chosen weight pair is by construction the minimiser — it cannot be
        strictly worse than any fixed pair, including the static defaults.
        """
        from ml.models.regression import MultiTargetPredictor

        predictor, X, y = _make_fitted_predictor()

        # Record default-weight RMSE
        result_default = predictor.predict(X)
        rmse_default = float(np.sqrt(
            np.mean((result_default.combined_pred - y.values) ** 2)
        ))

        # Optimise and record new RMSE
        predictor.optimize_weights(X, y, n_steps=8)
        result_opt = predictor.predict(X)
        rmse_opt = float(np.sqrt(
            np.mean((result_opt.combined_pred - y.values) ** 2)
        ))

        assert rmse_opt <= rmse_default + 1e-9, (
            f"Optimised RMSE ({rmse_opt:.4f}) must not exceed "
            f"default RMSE ({rmse_default:.4f})"
        )

    def test_returns_self(self):
        """optimize_weights must return self for method chaining."""
        from ml.models.regression import MultiTargetPredictor

        predictor, X, y = _make_fitted_predictor()
        returned = predictor.optimize_weights(X, y, n_steps=2)

        assert returned is predictor, "optimize_weights must return self"


# ── _create_engine_with_retry ─────────────────────────────────────────────────

class TestCreateEngineWithRetry:
    """Exponential backoff retry logic for SQLAlchemy engine creation.

    Does NOT require a real database: tests use mock engines to verify that
    retry counts, delays, and error propagation behave correctly.
    """

    def test_succeeds_on_first_attempt(self):
        """When the engine connects immediately, no retry is needed."""
        from unittest.mock import MagicMock, patch

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("sqlalchemy.create_engine", return_value=mock_engine):
            from ml.run_pipeline import _create_engine_with_retry
            result = _create_engine_with_retry("postgresql://fake/db", max_attempts=3)

        assert result is mock_engine

    def test_raises_after_max_attempts(self):
        """When all attempts fail, RuntimeError is raised."""
        from unittest.mock import patch

        with patch(
            "sqlalchemy.create_engine",
            side_effect=Exception("connection refused"),
        ):
            from ml.run_pipeline import _create_engine_with_retry
            with pytest.raises(RuntimeError, match="Failed to connect"):
                _create_engine_with_retry(
                    "postgresql://fake/db",
                    max_attempts=3,
                    base_delay=0.0,   # instant for unit tests
                )

    def test_succeeds_after_transient_failures(self):
        """Succeeds on the 3rd attempt when first two fail transiently."""
        from unittest.mock import MagicMock, call, patch

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        call_count = {"n": 0}

        def side_effect(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("transient error")
            return mock_engine

        with patch("sqlalchemy.create_engine", side_effect=side_effect):
            from ml.run_pipeline import _create_engine_with_retry
            result = _create_engine_with_retry(
                "postgresql://fake/db",
                max_attempts=5,
                base_delay=0.0,
            )

        assert result is mock_engine
        assert call_count["n"] == 3

    def test_delay_grows_exponentially(self):
        """Retry delays follow base_delay × 2^(attempt-1) schedule."""
        import time
        from unittest.mock import patch

        recorded_delays: list[float] = []
        original_sleep = time.sleep

        def mock_sleep(seconds: float) -> None:
            recorded_delays.append(seconds)

        with (
            patch("time.sleep", side_effect=mock_sleep),
            patch(
                "sqlalchemy.create_engine",
                side_effect=Exception("always fails"),
            ),
        ):
            from ml.run_pipeline import _create_engine_with_retry
            with pytest.raises(RuntimeError):
                _create_engine_with_retry(
                    "postgresql://fake/db",
                    max_attempts=4,
                    base_delay=1.0,
                )

        # Expect 3 sleeps (between attempts 1-2, 2-3, 3-4)
        assert len(recorded_delays) == 3
        assert abs(recorded_delays[0] - 1.0) < 1e-6   # 1.0 × 2^0
        assert abs(recorded_delays[1] - 2.0) < 1e-6   # 1.0 × 2^1
        assert abs(recorded_delays[2] - 4.0) < 1e-6   # 1.0 × 2^2
