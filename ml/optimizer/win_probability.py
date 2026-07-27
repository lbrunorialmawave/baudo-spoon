"""Monte Carlo win-probability estimate for a squad roster.

Parametric model: for each player, sample actual auction price from
Normal(effective_cost, std) where std = effective_cost * overpay_std_ratio.
Count fraction of simulations where total spend <= budget.

# ponytail: parametric fallback; swap for empirical distribution when
# auction history is sufficient to fit per-role price distributions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ml.optimizer.inflation import InflationConfig, compute_role_percentile_map, estimate_effective_cost
from ml.optimizer.models import Player

__all__ = ["WinProbabilityConfig", "estimate_completion_probability"]


@dataclass(frozen=True)
class WinProbabilityConfig:
    n_simulations: int = 1000
    overpay_std_ratio: float = 0.25


def estimate_completion_probability(
    squad: list[Player],
    budget: int,
    config: WinProbabilityConfig,
    inflation_config: InflationConfig,
    num_participants: int,
) -> float:
    """Return P(total auction cost <= budget) via Monte Carlo sampling.

    Each player's auction price is drawn from
    Normal(mu=effective_cost, sigma=mu * overpay_std_ratio), clamped to [1, inf).
    """
    if not squad:
        return 1.0

    percentiles = compute_role_percentile_map(squad)
    means: list[float] = [
        estimate_effective_cost(p, percentiles[p.player_id], num_participants, inflation_config)
        for p in squad
    ]
    stds: list[float] = [max(0.5, m * config.overpay_std_ratio) for m in means]

    wins = 0
    for _ in range(config.n_simulations):
        total = sum(
            max(1.0, random.gauss(mu, sigma))
            for mu, sigma in zip(means, stds)
        )
        if total <= budget:
            wins += 1
    return wins / config.n_simulations


if __name__ == "__main__":
    # quick self-check
    from ml.optimizer.models import Player
    p = Player(player_id="x", name="X", role="A", real_team="Inter", cost=20, projected_score=8.0)
    prob = estimate_completion_probability([p], budget=25, config=WinProbabilityConfig(n_simulations=500), inflation_config=InflationConfig(), num_participants=8)
    assert 0.0 <= prob <= 1.0, f"expected [0,1], got {prob}"
    print(f"OK: win_prob={prob:.2f}")
