"""Pareto frontier over (score, risk, auction feasibility) — Fase 4.2.

Per §7 del piano, questo modulo NON sostituisce le 4 strategie discrete
(scope esplicitamente escluso); è una vista aggiuntiva, opt-in, per chi vuole
esplorare il trade-off score/robustezza/fattibilità senza dover interpretare
4 rose indipendenti. Riusa meccanismi già esistenti — nessun nuovo solver:

- asse "score": objective medio della rosa (``total_projected_score``, non
  risk-adjusted, per essere confrontabile punto a punto)
- asse "risk": deviazione standard di portafoglio della rosa, stimata come
  sqrt(Σ prediction_std_i²) assumendo indipendenza tra giocatori (proxy, non
  correlazione reale — coerente con l'assenza di dati di covarianza nel
  progetto)
- asse "win_probability": riusa ``win_probability.estimate_completion_probability``
  (P(spesa asta <= budget))

Il frontier è generato risolvendo un piccolo grid di ``risk_aversion`` con il
solver ILP esistente (stesso meccanismo di Fase 0), poi filtrando i punti
non-dominati.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ml.optimizer.models import OptimizationConfig, Player, StrategyProfile
from ml.optimizer.optimizer import optimize_squad
from ml.optimizer.win_probability import (
    WinProbabilityConfig,
    estimate_completion_probability,
)

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RISK_LAMBDAS",
    "ParetoPoint",
    "ParetoResult",
    "compute_pareto_frontier",
]

DEFAULT_RISK_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


@dataclass
class ParetoPoint:
    risk_lambda: float
    status: str
    squad_ids: frozenset[str]
    score: float
    risk: float
    win_probability: float | None
    dominated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_lambda": self.risk_lambda,
            "status": self.status,
            "score": round(self.score, 3),
            "risk": round(self.risk, 4),
            "win_probability": round(self.win_probability, 4)
            if self.win_probability is not None
            else None,
            "squad_size": len(self.squad_ids),
            "dominated": self.dominated,
        }


@dataclass
class ParetoResult:
    points: list[ParetoPoint] = field(default_factory=list)
    frontier: list[ParetoPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "frontier_risk_lambdas": [p.risk_lambda for p in self.frontier],
            "warnings": list(self.warnings),
        }


def _portfolio_risk(squad: Sequence[Player]) -> float:
    stds = [
        p.prediction_std
        for p in squad
        if p.prediction_std is not None and p.prediction_std > 0
    ]
    if not stds:
        return 0.0
    return sum(s * s for s in stds) ** 0.5


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True if *a* dominates *b*: at least as good on every axis, strictly better on one."""
    wp_a = a.win_probability if a.win_probability is not None else 0.0
    wp_b = b.win_probability if b.win_probability is not None else 0.0
    at_least_as_good = a.score >= b.score and a.risk <= b.risk and wp_a >= wp_b
    strictly_better = a.score > b.score or a.risk < b.risk or wp_a > wp_b
    return at_least_as_good and strictly_better


def compute_pareto_frontier(
    pool: Sequence[Player],
    config: OptimizationConfig,
    strategy: StrategyProfile,
    *,
    risk_lambdas: Sequence[float] = DEFAULT_RISK_LAMBDAS,
    win_probability_config: WinProbabilityConfig | None = None,
    compute_win_probability: bool = True,
) -> ParetoResult:
    """Sweep ``risk_aversion`` and return every solved point plus the non-dominated frontier."""
    warnings: list[str] = []
    has_std = any(p.prediction_std is not None and p.prediction_std > 0 for p in pool)
    if not has_std:
        warnings.append(
            "no player carries prediction_std: risk axis will be 0 for every point (no differentiation)"
        )

    wp_cfg = win_probability_config or WinProbabilityConfig()
    points: list[ParetoPoint] = []
    for lam in risk_lambdas:
        variant_cfg = replace(config, risk_aversion=lam)
        try:
            res = optimize_squad(list(pool), variant_cfg, strategy)
        except Exception as exc:
            warnings.append(f"risk_lambda={lam} failed: {exc}")
            continue
        if not res.squad:
            points.append(ParetoPoint(lam, res.status, frozenset(), 0.0, 0.0, None))
            continue
        wp = None
        if compute_win_probability:
            try:
                wp = estimate_completion_probability(
                    res.squad,
                    config.budget,
                    wp_cfg,
                    config.inflation_config,
                    config.num_participants,
                )
            except Exception as exc:
                warnings.append(f"win_probability failed for risk_lambda={lam}: {exc}")
        points.append(
            ParetoPoint(
                risk_lambda=lam,
                status=res.status,
                squad_ids=frozenset(sp.player_id for sp in res.squad),
                score=res.total_projected_score,
                risk=_portfolio_risk(res.squad),
                win_probability=wp,
            )
        )

    # Non-dominated frontier — O(n^2), n is tiny (len(risk_lambdas)).
    frontier: list[ParetoPoint] = []
    for p in points:
        if not p.squad_ids:
            continue
        if any(_dominates(q, p) for q in points if q is not p and q.squad_ids):
            p.dominated = True
        else:
            frontier.append(p)

    return ParetoResult(points=points, frontier=frontier, warnings=warnings)
