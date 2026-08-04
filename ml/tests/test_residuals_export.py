"""Walk-forward residuals export for optimizer Monte Carlo."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluation.metrics import backtest
from ml.evaluation.residuals_export import build_residuals_payload, summarize_residuals
from ml.optimizer.residual_loader import load_residuals_from_path


def _toy_df(n_seasons: int = 3, n_per: int = 20) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for s in range(2020, 2020 + n_seasons):
        for i in range(n_per):
            x = float(rng.normal())
            rows.append(
                {
                    "player_fotmob_id": f"fm-{i % 10}",
                    "canonical_role": ["P", "D", "C", "A"][i % 4],
                    "season_start": s,
                    "feat": x,
                    "fantavoto_medio": 5.0 + 0.5 * x + float(rng.normal(0, 0.3)),
                }
            )
    return pd.DataFrame(rows)


def test_backtest_collects_residuals():
    df = _toy_df()
    pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    bt = backtest(pipe, df, feature_cols=["feat"], model_name="toy")
    assert bt.season_metrics, "expected at least one test season"
    assert bt.residuals, "expected residual rows"
    row = bt.residuals[0]
    assert "player_id" in row and "role" in row and "residual" in row
    assert abs(row["residual"] - (row["actual"] - row["predicted"])) < 1e-9


def test_payload_and_loader_roundtrip(tmp_path: Path):
    df = _toy_df()
    pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    bt = backtest(pipe, df, feature_cols=["feat"], model_name="toy")
    payload = build_residuals_payload(bt.residuals, run_id="test-run", model_name="toy")
    assert payload["schema_version"] == 1
    assert payload["n_rows"] == len(bt.residuals)
    path = tmp_path / "residuals.json"
    path.write_text(json.dumps(payload))
    report = load_residuals_from_path(path)
    assert report.residuals
    assert report.n_players >= 1
    stats = summarize_residuals(bt.residuals)
    assert stats["n_rows"] > 0
    assert stats["mean_abs_residual"] is not None
