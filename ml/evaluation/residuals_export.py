"""Export walk-forward prediction residuals for optimizer Monte Carlo.

Schema (written as ``residuals.json`` via ArtifactStore → local + R2):

```json
{
  "schema_version": 1,
  "source": "walkforward_backtest",
  "run_id": "...",
  "model_name": "...",
  "n_rows": 1234,
  "n_players": 400,
  "n_roles": 4,
  "residuals": [
    {"player_id": "fm-123", "role": "A", "residual": 0.42,
     "actual": 6.5, "predicted": 6.08, "season_start": 2024}
  ]
}
```

``residual = actual_fantavoto_medio - predicted``.
Compatible with ``ml.optimizer.residual_loader`` and ``MonteCarloSimulator.fit``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def build_residuals_payload(
    residuals: Sequence[dict[str, Any]],
    *,
    run_id: str = "",
    model_name: str = "",
    source: str = "walkforward_backtest",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap residual rows in the artifact envelope."""
    players = {
        str(r.get("player_id")) for r in residuals if r.get("player_id") is not None
    }
    roles = {str(r.get("role")) for r in residuals if r.get("role") is not None}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "run_id": run_id,
        "model_name": model_name,
        "n_rows": len(residuals),
        "n_players": len(players),
        "n_roles": len(roles),
        "residuals": list(residuals),
    }
    if extra_meta:
        payload["meta"] = extra_meta
    return payload


def summarize_residuals(residuals: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight stats for logs / diagnostics (no full residual dump)."""
    if not residuals:
        return {"n_rows": 0, "mean_abs_residual": None, "rmse_residual": None}
    vals = [float(r["residual"]) for r in residuals if r.get("residual") is not None]
    if not vals:
        return {"n_rows": 0, "mean_abs_residual": None, "rmse_residual": None}
    import math

    n = len(vals)
    mean_abs = sum(abs(v) for v in vals) / n
    rmse = math.sqrt(sum(v * v for v in vals) / n)
    return {
        "n_rows": n,
        "n_players": len({str(r.get("player_id")) for r in residuals}),
        "mean_abs_residual": round(mean_abs, 4),
        "rmse_residual": round(rmse, 4),
    }
