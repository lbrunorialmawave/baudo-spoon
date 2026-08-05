"""Structured diagnostics for optimizer runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ml.optimizer.models import OptimizationResult, Player

__all__ = ["PoolDiagnostics", "build_pool_diagnostics", "merge_result_diagnostics"]


@dataclass
class PoolDiagnostics:
    pool_size: int
    pct_with_prediction_std: float
    pct_with_season_value: float
    pct_with_start_probability: float
    cost_p50: float
    cost_p90: float
    projected_score_p50: float
    projected_score_p90: float
    roles: dict[str, int] = field(default_factory=dict)
    pct_with_var_score: float = 0.0
    pct_with_esv: float = 0.0
    mean_prediction_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "pct_with_prediction_std": round(self.pct_with_prediction_std, 4),
            "pct_with_season_value": round(self.pct_with_season_value, 4),
            "pct_with_start_probability": round(self.pct_with_start_probability, 4),
            "pct_with_var_score": round(self.pct_with_var_score, 4),
            "pct_with_esv": round(self.pct_with_esv, 4),
            "mean_prediction_std": (
                round(self.mean_prediction_std, 4) if self.mean_prediction_std is not None else None
            ),
            "cost_p50": self.cost_p50,
            "cost_p90": self.cost_p90,
            "projected_score_p50": round(self.projected_score_p50, 3),
            "projected_score_p90": round(self.projected_score_p90, 3),
            "roles": dict(self.roles),
        }


def build_pool_diagnostics(pool: Sequence[Player]) -> PoolDiagnostics:
    n = len(pool)
    if n == 0:
        return PoolDiagnostics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    costs = np.asarray([p.cost for p in pool], dtype=float)
    scores = np.asarray([p.projected_score for p in pool], dtype=float)
    roles: dict[str, int] = {}
    stds: list[float] = []
    for p in pool:
        roles[p.role] = roles.get(p.role, 0) + 1
        if p.prediction_std is not None and p.prediction_std > 0:
            stds.append(float(p.prediction_std))
    return PoolDiagnostics(
        pool_size=n,
        pct_with_prediction_std=sum(
            1 for p in pool if p.prediction_std is not None and p.prediction_std > 0
        )
        / n,
        pct_with_season_value=sum(
            1 for p in pool if p.season_value is not None and p.season_value > 0
        )
        / n,
        pct_with_start_probability=sum(1 for p in pool if p.start_probability is not None) / n,
        cost_p50=float(np.percentile(costs, 50)),
        cost_p90=float(np.percentile(costs, 90)),
        projected_score_p50=float(np.percentile(scores, 50)),
        projected_score_p90=float(np.percentile(scores, 90)),
        roles=roles,
        pct_with_var_score=sum(1 for p in pool if getattr(p, "var_score", None) is not None) / n,
        pct_with_esv=sum(1 for p in pool if getattr(p, "esv", None) is not None) / n,
        mean_prediction_std=float(np.mean(stds)) if stds else None,
    )


def merge_result_diagnostics(result, pool_diag, *, extra=None):
    merged = dict(result.diagnostics) if result.diagnostics else {}
    merged["pool"] = pool_diag.to_dict() if hasattr(pool_diag, "to_dict") else pool_diag
    if extra:
        merged.update(extra)
    return merged
