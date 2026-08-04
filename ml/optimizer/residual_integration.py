"""Helpers to build MC simulator preferring evaluation residuals."""
from __future__ import annotations

import os
from typing import Sequence

from ml.optimizer.models import Player
from ml.optimizer.monte_carlo_opt import build_simulator_from_pool, build_simulator_from_residuals
from ml.optimizer.residual_loader import load_residuals_from_artifacts, merge_with_prediction_std
from ml.simulations.monte_carlo import MonteCarloSimulator


def build_simulator_preferring_residuals(
    pool: Sequence[Player],
    *,
    random_seed: int = 42,
    artifacts_dir: str | None = None,
) -> tuple[MonteCarloSimulator, list[str], dict]:
    """Prefer disk residuals; merge prediction_std; else pure prediction_std path."""
    warnings: list[str] = []
    art = artifacts_dir or os.environ.get("API_ARTIFACTS_DIR") or os.environ.get("ARTIFACTS_DIR")
    report = load_residuals_from_artifacts(art)
    meta = report.to_dict()
    if report.residuals:
        merged = merge_with_prediction_std(report.residuals, pool, random_seed=random_seed)
        sim = build_simulator_from_residuals(merged, random_seed=random_seed)
        warnings.extend(report.warnings)
        warnings.append(f"using residuals source={report.source} rows={len(merged)}")
        meta["merged_rows"] = len(merged)
        return sim, warnings, meta
    sim, w = build_simulator_from_pool(pool, random_seed=random_seed)
    warnings.extend(w)
    warnings.extend(report.warnings)
    return sim, warnings, meta
