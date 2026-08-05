"""Helpers to build MC simulator preferring evaluation residuals."""
from __future__ import annotations

import os
from typing import Any, Sequence

from ml.optimizer.models import Player
from ml.optimizer.monte_carlo_opt import build_simulator_from_pool, build_simulator_from_residuals
from ml.optimizer.residual_loader import (
    load_residuals_from_artifacts,
    load_residuals_preferring_r2,
    merge_with_prediction_std,
)
from ml.simulations.monte_carlo import MonteCarloSimulator


def build_simulator_preferring_residuals(
    pool: Sequence[Player],
    *,
    random_seed: int = 42,
    artifacts_dir: str | None = None,
    r2_endpoint_url: str | None = None,
    r2_access_key_id: str | None = None,
    r2_secret_access_key: str | None = None,
    r2_bucket_name: str | None = None,
) -> tuple[MonteCarloSimulator, list[str], dict[str, Any]]:
    """Prefer residual file (local → R2); merge prediction_std; else pool-only path.

    Returns:
        (simulator, warnings, meta) where meta includes residual source diagnostics
        for API ``monteCarloSummary`` / result diagnostics.
    """
    warnings: list[str] = []
    art = artifacts_dir or os.environ.get("API_ARTIFACTS_DIR") or os.environ.get("ARTIFACTS_DIR")

    # Prefer R2-aware loader when credentials might exist
    report = load_residuals_preferring_r2(
        art,
        r2_endpoint_url=r2_endpoint_url,
        r2_access_key_id=r2_access_key_id,
        r2_secret_access_key=r2_secret_access_key,
        r2_bucket_name=r2_bucket_name,
    )
    meta: dict[str, Any] = report.to_dict()
    meta["residual_source"] = report.source

    if report.residuals:
        merged = merge_with_prediction_std(report.residuals, pool, random_seed=random_seed)
        sim = build_simulator_from_residuals(merged, random_seed=random_seed)
        warnings.extend(report.warnings)
        warnings.append(
            f"using residuals source={report.source} rows={len(report.residuals)} merged={len(merged)}"
        )
        meta["merged_rows"] = len(merged)
        meta["using"] = "walkforward_residuals"
        return sim, warnings, meta

    # Fallback pure prediction_std / parametric
    sim, w = build_simulator_from_pool(pool, random_seed=random_seed)
    warnings.extend(w)
    warnings.extend(report.warnings)
    meta["using"] = "prediction_std_or_parametric"
    meta["merged_rows"] = 0
    return sim, warnings, meta
