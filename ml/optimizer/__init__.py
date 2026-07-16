"""Fantacalcio squad optimizer (ILP-based).

Public surface:

* :class:`Player`, :class:`Formation`, :class:`InflationConfig`,
  :class:`StrategyProfile`, :class:`OptimizationConfig`,
  :class:`OptimizationResult`, :class:`MultiStrategyResult` - data models.
* :func:`estimate_effective_cost` - pure function modelling auction inflation.
* :data:`DEFAULT_FOUR_STRATEGIES` - the 4 default strategy profiles.
* :func:`optimize_squad` - single-strategy ILP entry point.
* :func:`optimize_multi_strategy` - run the 4 strategies independently.
"""

from __future__ import annotations

from ml.optimizer.inflation import estimate_effective_cost
from ml.optimizer.models import (
    Formation,
    InflationConfig,
    MultiStrategyResult,
    OptimizationConfig,
    OptimizationResult,
    Player,
    Role,
    StrategyProfile,
)
from ml.optimizer.optimizer import optimize_multi_strategy, optimize_squad
from ml.optimizer.strategies import DEFAULT_FOUR_STRATEGIES, default_strategies

__all__ = [
    "DEFAULT_FOUR_STRATEGIES",
    "Formation",
    "InflationConfig",
    "MultiStrategyResult",
    "OptimizationConfig",
    "OptimizationResult",
    "Player",
    "Role",
    "StrategyProfile",
    "default_strategies",
    "estimate_effective_cost",
    "optimize_multi_strategy",
    "optimize_squad",
]

