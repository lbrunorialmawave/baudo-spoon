"""Test that --evaluate-mantra in run_pipeline actually calls evaluate_mantra_vs_actuals.

Process guard: proves the function is reachable from the CLI entry point.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestEvaluateMantraWiring:
    """Task 2: pipeline CLI calls evaluate_mantra_vs_actuals when flag is set."""

    def test_evaluate_mantra_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ML_DATABASE_URL", "postgresql://fake:5432/test")

        # Mock heavy dependencies that require DB / pydantic-settings
        mock_config = MagicMock()
        mock_config.log_level = "WARNING"
        mock_config.artifacts_dir = "/tmp/artifacts"
        mock_config.predict_next = False
        mock_config.random_seed = 42

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.run.return_value = {
            "best_model": "xgb",
            "predictions": [{"season_start": 2024}],
            "model_comparison": [{"model": "xgb", "rmse": 0.5}],
            "backtest": {"mean_rmse": 0.5, "mean_mae": 0.4, "mean_r2": 0.8},
            "clustering_stats": {"n_clusters": 6, "silhouette": 0.4},
            "low_cost_recommendations": [],
            "metadata": {"config": {"season_start": 2024}},
            "config": {"test_seasons": 1},
        }

        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_ml_config_cls = MagicMock(return_value=mock_config)

        mock_run_mantra = MagicMock(
            return_value={"players": [{"fantacalcio_id": 1, "VR": 6.5}]}
        )
        mock_evaluate = MagicMock(return_value={"rmse": 0.8, "mae": 0.6, "r2": 0.5})
        mock_engine = MagicMock()

        # Simulate CLI args
        test_args = [
            "ml.run_pipeline",
            "--evaluate-mantra",
            "--log-level",
            "WARNING",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        with (
            patch(
                "ml.run_pipeline._create_engine_with_retry", return_value=mock_engine
            ),
            patch.dict(
                "sys.modules",
                {
                    "ml.config": MagicMock(MLConfig=mock_ml_config_cls),
                    "ml.pipeline.trainer": MagicMock(Trainer=mock_trainer_cls),
                },
            ),
            patch("ml.mantra.runner.run_mantra", mock_run_mantra),
            patch("ml.mantra.evaluate.evaluate_mantra_vs_actuals", mock_evaluate),
        ):
            from ml.run_pipeline import main

            exit_code = main()

        assert exit_code == 0
        mock_run_mantra.assert_called_once_with(mock_engine, 2024)
        mock_evaluate.assert_called_once_with(
            mock_run_mantra.return_value, mock_engine, 2024
        )
