"""Canonical decision-score policy for Auction and secondary paths.

Single source of truth for the decision-layer transform:

    projected_score
      → reliability_weight multiplier (when apply_reliability_weight)
      → risk_aversion * prediction_std penalty
      → decision_score

Used by VarEngine, alternatives ranking, and Monte Carlo when the purpose
is a *decision* (selection / ranking), not pure display.

See ADR 0001 and docs/config/auction-reliability-contract.md.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

__all__ = [
    "DEFAULT_APPLY_RELIABILITY_WEIGHT",
    "DEFAULT_RISK_AVERSION",
    "DEFAULT_RELIABILITY_WEIGHT_MODE",
    "compute_decision_score",
    "compute_decision_score_from_player",
]


# Single source of truth defaults (WS1)
DEFAULT_APPLY_RELIABILITY_WEIGHT: bool = True
DEFAULT_RISK_AVERSION: float = 0.0
DEFAULT_RELIABILITY_WEIGHT_MODE: str = "continuous"


class _HasScoreFields(Protocol):
    projected_score: float
    reliability_weight: float | None
    prediction_std: float | None
    season_value: float | None


def compute_decision_score(
    *,
    projected_score: float,
    reliability_weight: float | None = None,
    prediction_std: float | None = None,
    apply_reliability_weight: bool = DEFAULT_APPLY_RELIABILITY_WEIGHT,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    season_value: float | None = None,
    use_season_value: bool = False,
) -> float:
    """Compute the canonical decision score.

    Args:
        projected_score: Display/model projected score (already display-shrunk
            when output reliability is attached).
        reliability_weight: Decision-layer weight in [floor, 1]. Ignored when
            ``apply_reliability_weight`` is False or value is None/negative.
        prediction_std: Ensemble std; used only when ``risk_aversion > 0``.
        apply_reliability_weight: Master switch (default True per ADR 0001).
        risk_aversion: Penalty multiplier (default 0.0, opt-in).
        season_value: Optional season-value alternative base score.
        use_season_value: When True and season_value is valid, use it as base.

    Returns:
        Decision score (float). Never raises on missing optional fields.
    """
    if use_season_value and isinstance(season_value, (int, float)) and season_value > 0:
        base = float(season_value)
    else:
        base = float(projected_score)

    if apply_reliability_weight:
        if isinstance(reliability_weight, (int, float)) and reliability_weight >= 0:
            base = base * float(reliability_weight)

    if risk_aversion > 0.0:
        if isinstance(prediction_std, (int, float)) and prediction_std >= 0:
            base = base - float(risk_aversion) * float(prediction_std)

    return base


def compute_decision_score_from_player(
    player: _HasScoreFields | Mapping[str, Any] | Any,
    *,
    apply_reliability_weight: bool = DEFAULT_APPLY_RELIABILITY_WEIGHT,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    use_season_value: bool = False,
) -> float:
    """Convenience wrapper accepting Player dataclass, dict, or duck-typed object."""
    if isinstance(player, Mapping):
        projected = float(player.get("projected_score") or 0.0)
        rw = player.get("reliability_weight")
        std = player.get("prediction_std")
        sv = player.get("season_value")
    else:
        projected = float(getattr(player, "projected_score", 0.0) or 0.0)
        rw = getattr(player, "reliability_weight", None)
        std = getattr(player, "prediction_std", None)
        sv = getattr(player, "season_value", None)

    return compute_decision_score(
        projected_score=projected,
        reliability_weight=rw if isinstance(rw, (int, float)) else None,
        prediction_std=std if isinstance(std, (int, float)) else None,
        apply_reliability_weight=apply_reliability_weight,
        risk_aversion=risk_aversion,
        season_value=sv if isinstance(sv, (int, float)) else None,
        use_season_value=use_season_value,
    )
