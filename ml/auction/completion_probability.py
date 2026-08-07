"""Live completion-probability indicator for auction participants (WS3 #1).

Reuses the Monte Carlo model from :mod:`ml.optimizer.win_probability` but
applies it to the *remaining* roster of a participant mid-auction:

* residual budget (instead of full budget)
* residual role quotas still to fill
* expected prices of the cheapest available players per residual slot
  (proxy for "what it will cost to finish the roster")

This is pure domain logic — no FastAPI dependency — so it can be called
from the orchestrator summary path or the API layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ml.auction.models import AuctionState, ParticipantState
from ml.auction.price_drift import project_price_for_player
from ml.optimizer.models import Player
from ml.optimizer.win_probability import WinProbabilityConfig

__all__ = [
    "CompletionProbabilityConfig",
    "estimate_all_completion_probabilities",
    "estimate_participant_completion_probability",
]


@dataclass(frozen=True)
class CompletionProbabilityConfig:
    """Thin wrapper so auction callers need not import optimizer config."""

    n_simulations: int = 500
    overpay_std_ratio: float = 0.25

    def to_win_config(self) -> WinProbabilityConfig:
        return WinProbabilityConfig(
            n_simulations=self.n_simulations,
            overpay_std_ratio=self.overpay_std_ratio,
        )


def _remaining_slots(
    participant: ParticipantState, role_quotas: dict[str, int]
) -> dict[str, int]:
    """Residual role quotas still to fill for this participant."""
    out: dict[str, int] = {}
    for role, quota in role_quotas.items():
        filled = participant.role_breakdown.get(role, 0)
        residual = quota - filled
        if residual > 0:
            out[role] = residual
    return out


def _slot_expected_costs(
    state: AuctionState,
    remaining: dict[str, int],
) -> list[float]:
    """Pick expected prices for residual slots from the available pool.

    For each residual slot of role R, take the *cheapest* still-available
    players compatible with R (by classic ``role`` or MANTRA
    ``eligible_roles``). If the pool is short, fall back to the median
    expected price of the role; if the role is empty, use 1.0 (minimum bid).
    """
    ruleset = getattr(state.config, "ruleset", "CLASSIC") or "CLASSIC"
    costs: list[float] = []

    # Pre-compute expected prices once.
    price_by_id: dict[str, float] = {
        p.player_id: max(1.0, project_price_for_player(state, p))
        for p in state.available_pool
    }

    def _covers(p: Player, role: str) -> bool:
        if ruleset == "MANTRA" and p.eligible_roles:
            return role in p.eligible_roles
        return p.role == role

    for role, n_slots in remaining.items():
        candidates = sorted(
            (
                price_by_id[p.player_id]
                for p in state.available_pool
                if _covers(p, role)
            ),
        )
        if not candidates:
            costs.extend([1.0] * n_slots)
            continue
        # Use the n cheapest; if fewer than n, repeat the most expensive of them.
        for i in range(n_slots):
            if i < len(candidates):
                costs.append(candidates[i])
            else:
                costs.append(candidates[-1])
    return costs


def estimate_participant_completion_probability(
    state: AuctionState,
    participant_id: str,
    config: CompletionProbabilityConfig | None = None,
) -> float:
    """P(participant can finish their roster within residual budget).

    Returns 1.0 when the roster is already complete, 0.0 when residual
    budget is exhausted with slots still open and no free (cost=0) path.
    """
    cfg = config or CompletionProbabilityConfig()
    participant = state.participants.get(participant_id)
    if participant is None:
        raise KeyError(f"unknown participant_id {participant_id!r}")

    remaining = _remaining_slots(participant, state.config.role_quotas)
    if not remaining:
        return 1.0

    means = _slot_expected_costs(state, remaining)
    if not means:
        return 1.0

    budget = max(0, participant.budget_residual)
    # Credit-reserve rule: after filling all remaining slots, at least 0 left
    # is acceptable (the orchestrator enforces >= 1 per remaining-before-buy;
    # here we model the final fill).
    stds = [max(0.5, m * cfg.overpay_std_ratio) for m in means]
    wins = 0
    for _ in range(cfg.n_simulations):
        total = sum(max(1.0, random.gauss(mu, sigma)) for mu, sigma in zip(means, stds))
        if total <= budget:
            wins += 1
    return wins / cfg.n_simulations


def estimate_all_completion_probabilities(
    state: AuctionState,
    config: CompletionProbabilityConfig | None = None,
) -> dict[str, float]:
    """Map participant_id → completion probability for the whole session."""
    return {
        pid: estimate_participant_completion_probability(state, pid, config)
        for pid in state.participants
    }
