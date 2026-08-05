"""Sensitivity matrix — "se muovo X, cosa cambia in media sulle rose" (Fase 4.5).

Pure domain: given a baseline pool/config/strategy, resolves a small grid of
variants for each of a handful of objective/budget knobs (risk_aversion,
var_blend, hybrid_blend, budget) and reports, per variant, how much the squad
and score moved relative to the baseline solve. No FastAPI here — the router
translates this into I/O, same convention as the rest of ml/optimizer.

This does not attempt exhaustive multi-parameter sensitivity (that's a
combinatorial grid best done offline); it perturbs one parameter at a time
around the baseline config, holding everything else fixed — a classic
one-at-a-time (OAT) sensitivity analysis, cheap enough to run inline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ml.optimizer.diversity import _jaccard
from ml.optimizer.models import OptimizationConfig, Player, StrategyProfile
from ml.optimizer.optimizer import optimize_squad

log = logging.getLogger(__name__)

__all__ = [
    "SensitivityPoint",
    "ParameterSensitivity",
    "SensitivityResult",
    "compute_sensitivity_matrix",
    "DEFAULT_GRIDS",
]

# Default one-at-a-time grids. Each is applied independently on top of the
# baseline config; budget is expressed as a multiplier so it composes with
# any budget the caller passes in.
DEFAULT_GRIDS: dict[str, list[float]] = {
    "risk_aversion": [0.0, 0.5, 1.0],
    "var_blend": [0.0, 0.25, 0.5, 1.0],
    "hybrid_blend": [0.0, 0.25, 0.5, 1.0],
    "budget_multiplier": [0.9, 0.95, 1.0, 1.05, 1.1],
}


@dataclass
class SensitivityPoint:
    value: float
    status: str
    squad_ids: frozenset[str]
    total_score: float
    score_delta: float
    score_delta_pct: float
    jaccard_vs_baseline: float
    players_changed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "total_score": round(self.total_score, 3),
            "score_delta": round(self.score_delta, 3),
            "score_delta_pct": round(self.score_delta_pct, 4),
            "jaccard_vs_baseline": round(self.jaccard_vs_baseline, 4),
            "players_changed": self.players_changed,
        }


@dataclass
class ParameterSensitivity:
    parameter: str
    points: list[SensitivityPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"parameter": self.parameter, "points": [p.to_dict() for p in self.points]}


@dataclass
class SensitivityResult:
    baseline_status: str
    baseline_total_score: float
    baseline_squad_ids: frozenset[str]
    parameters: list[ParameterSensitivity] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_status": self.baseline_status,
            "baseline_total_score": round(self.baseline_total_score, 3),
            "baseline_squad_size": len(self.baseline_squad_ids),
            "parameters": [p.to_dict() for p in self.parameters],
            "warnings": list(self.warnings),
        }


def _apply(config: OptimizationConfig, param: str, value: float) -> OptimizationConfig:
    if param == "budget_multiplier":
        return replace(config, budget=max(1, round(config.budget * value)))
    return replace(config, **{param: value})


def compute_sensitivity_matrix(
    pool: Sequence[Player],
    config: OptimizationConfig,
    strategy: StrategyProfile,
    *,
    grids: dict[str, list[float]] | None = None,
) -> SensitivityResult:
    """Run the OAT sensitivity sweep and return per-parameter deltas vs baseline.

    Parameters with no meaningful signal in the pool are skipped with a warning
    rather than producing a flat/degenerate row (e.g. ``var_blend`` when no
    player in the pool has ``var_score`` populated).
    """
    grids = grids or DEFAULT_GRIDS
    warnings: list[str] = []

    try:
        baseline = optimize_squad(list(pool), config, strategy)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"baseline solve failed, cannot compute sensitivity: {exc}") from exc

    if not baseline.squad:
        return SensitivityResult(baseline.status, 0.0, frozenset(), [], [f"baseline status={baseline.status}, no squad"])

    baseline_ids = frozenset(sp.player_id for sp in baseline.squad)
    baseline_score = baseline.total_projected_score

    availability = {
        "var_blend": any(getattr(p, "var_score", None) is not None for p in pool),
        "hybrid_blend": any(getattr(p, "fp_ibrido", None) is not None for p in pool),
        "risk_aversion": any(getattr(p, "prediction_std", None) for p in pool),
        "budget_multiplier": True,
    }

    parameters: list[ParameterSensitivity] = []
    for param, values in grids.items():
        if not availability.get(param, True):
            warnings.append(f"skipped '{param}': no player in pool carries the underlying signal")
            continue
        points: list[SensitivityPoint] = []
        for value in values:
            variant_cfg = _apply(config, param, value)
            try:
                res = optimize_squad(list(pool), variant_cfg, strategy)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{param}={value} failed: {exc}")
                continue
            ids = frozenset(sp.player_id for sp in res.squad) if res.squad else frozenset()
            score = res.total_projected_score if res.squad else 0.0
            points.append(
                SensitivityPoint(
                    value=value,
                    status=res.status,
                    squad_ids=ids,
                    total_score=score,
                    score_delta=score - baseline_score,
                    score_delta_pct=(score - baseline_score) / baseline_score if baseline_score else 0.0,
                    jaccard_vs_baseline=_jaccard(ids, baseline_ids),
                    players_changed=len(baseline_ids ^ ids),
                )
            )
        parameters.append(ParameterSensitivity(param, points))

    return SensitivityResult(baseline.status, baseline_score, baseline_ids, parameters, warnings)
