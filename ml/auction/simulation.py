"""Monte Carlo simulation of a full auction with synthetic bidders.

Stateless engine: no interaction with in-memory auction sessions.
Reuses orchestrator state transitions and price_drift projections.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.auction.models import (
    AuctionConfig,
    AuctionState,
    ParticipantSetup,
    ParticipantState,
)
from ml.auction.orchestrator import initialize_auction, record_assignment
from ml.auction.price_drift import project_price_for_player
from ml.optimizer.models import Player

log = logging.getLogger(__name__)

__all__ = [
    "AuctionSimulationConfig",
    "AuctionSimulationResult",
    "BidderPolicy",
    "BidderProfile",
    "ParticipantSimStats",
    "simulate_auction",
]


def _clamp01(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"BidderPolicy.{name} must be in [0, 1], got {value}")
    return float(value)


@dataclass(frozen=True)
class BidderPolicy:
    aggressiveness: float = 0.5
    inflation_tolerance: float = 0.5
    max_overpay_ratio: float = 1.2
    min_residual_credits_per_slot: float = 1.5
    all_in_probability: float = 0.1
    budget_elasticity: float = 0.4
    var_weight: float = 0.35
    team_strength_weight: float = 0.15
    prefer_alternatives: bool = True
    prefer_low_cost_alternative: bool = False
    rebid_trigger_pct_above_expected: float = 0.12
    budget_share_by_role: Mapping[str, float] | None = None
    phase_bias: str | None = None
    prefer_young_players: bool = False
    max_age_preference: int | None = None
    prefer_high_start_probability: bool = False
    min_start_probability: float | None = None
    prefer_high_variance: bool = False
    prefer_multi_role: bool = False
    min_num_roles: int | None = None
    budget_share_by_block: Mapping[str, float] | None = None
    max_top_tier_count: int | None = None
    target_top_tier_count: int | None = None
    avoid_top_tier_early: bool = False
    adaptive: bool = False
    adapt_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _clamp01("aggressiveness", self.aggressiveness)
        _clamp01("inflation_tolerance", self.inflation_tolerance)
        _clamp01("all_in_probability", self.all_in_probability)
        _clamp01("budget_elasticity", self.budget_elasticity)
        _clamp01("var_weight", self.var_weight)
        _clamp01("team_strength_weight", self.team_strength_weight)
        if self.max_overpay_ratio < 1.0:
            raise ValueError(
                f"BidderPolicy.max_overpay_ratio must be >= 1.0, got {self.max_overpay_ratio}"
            )
        if self.min_residual_credits_per_slot < 0:
            raise ValueError("BidderPolicy.min_residual_credits_per_slot must be >= 0")
        if self.rebid_trigger_pct_above_expected < 0:
            raise ValueError(
                "BidderPolicy.rebid_trigger_pct_above_expected must be >= 0"
            )


@dataclass(frozen=True)
class BidderProfile:
    participant_id: str
    policy: BidderPolicy


@dataclass(frozen=True)
class AuctionSimulationConfig:
    n_simulations: int = 200
    random_seed: int = 42
    price_noise_std_ratio: float = 0.15
    timeout_seconds: float = 0.0
    min_bid_step: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.n_simulations <= 1000:
            raise ValueError(
                f"n_simulations must be in 1..1000, got {self.n_simulations}"
            )
        if self.price_noise_std_ratio < 0:
            raise ValueError("price_noise_std_ratio must be >= 0")
        if self.min_bid_step < 1:
            raise ValueError("min_bid_step must be >= 1")


@dataclass(frozen=True)
class ParticipantSimStats:
    spend_p10: float
    spend_p50: float
    spend_p90: float
    esv_total_p10: float
    esv_total_p50: float
    esv_total_p90: float
    completion_probability: float
    squad_composition_mode: dict[str, int]


@dataclass(frozen=True)
class AuctionSimulationResult:
    n_completed: int
    per_participant: dict[str, ParticipantSimStats]
    price_index_drift_p50: dict[str, dict[str, float]]
    player_acquisition_probability: dict[str, dict[str, float]]
    wall_time_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "n_completed": self.n_completed,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "warnings": list(self.warnings),
            "per_participant": {
                pid: {
                    "spend_p10": round(s.spend_p10, 2),
                    "spend_p50": round(s.spend_p50, 2),
                    "spend_p90": round(s.spend_p90, 2),
                    "esv_total_p10": round(s.esv_total_p10, 2),
                    "esv_total_p50": round(s.esv_total_p50, 2),
                    "esv_total_p90": round(s.esv_total_p90, 2),
                    "completion_probability": round(s.completion_probability, 4),
                    "squad_composition_mode": dict(s.squad_composition_mode),
                }
                for pid, s in self.per_participant.items()
            },
            "price_index_drift_p50": {
                role: {t: round(v, 4) for t, v in tiers.items()}
                for role, tiers in self.price_index_drift_p50.items()
            },
            "player_acquisition_probability": {
                pid: {
                    "prob": round(stats["prob"], 4),
                    "avg_price": round(stats["avg_price"], 2),
                }
                for pid, stats in self.player_acquisition_probability.items()
            },
        }


def _player_roles(player: Player, ruleset: str) -> list[str]:
    if ruleset == "MANTRA":
        if player.eligible_roles:
            return list(player.eligible_roles)
        return [player.role] if player.role else []
    return [player.role] if player.role else []


def _remaining_slots(
    participant: ParticipantState, role_quotas: Mapping[str, int]
) -> dict[str, int]:
    return {
        role: max(0, int(quota) - int(participant.role_breakdown.get(role, 0)))
        for role, quota in role_quotas.items()
    }


def _total_remaining_slots(remaining: Mapping[str, int]) -> int:
    return sum(remaining.values())


def _bot_bid(
    player: Player,
    bidder: ParticipantState,
    state: AuctionState,
    policy: BidderPolicy,
    rng: np.random.Generator,
    price_noise_std_ratio: float,
) -> float | None:
    config = state.config
    remaining = _remaining_slots(bidder, config.role_quotas)
    total_slots_left = _total_remaining_slots(remaining)
    if total_slots_left <= 0:
        return None
    roles = _player_roles(player, config.ruleset)
    if not [r for r in roles if remaining.get(r, 0) > 0]:
        return None
    if (
        policy.min_start_probability is not None
        and player.start_probability is not None
        and float(player.start_probability) < policy.min_start_probability
    ):
        return None
    expected = float(project_price_for_player(state, player))
    if expected <= 0:
        expected = float(player.cost or 1)
    reserve = policy.min_residual_credits_per_slot * max(0, total_slots_left - 1)
    max_affordable = max(0.0, float(bidder.budget_residual) - reserve)
    if max_affordable < 1:
        return None
    overpay = policy.max_overpay_ratio * (0.85 + 0.3 * policy.aggressiveness)
    noise = float(rng.normal(0.0, price_noise_std_ratio))
    raw_bid = expected * overpay * (1.0 + noise)
    if rng.random() < policy.all_in_probability:
        raw_bid *= 1.0 + 0.25 * policy.aggressiveness
    bid = min(raw_bid, max_affordable)
    return float(bid) if bid >= 1 else None


def _resolve_winner(
    bids: dict[str, float], min_step: int = 1
) -> tuple[str, float] | None:
    if not bids:
        return None
    ordered = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
    winner_id, top = ordered[0]
    if len(ordered) == 1:
        price = max(1.0, float(top))
    else:
        price = max(1.0, min(float(top), float(ordered[1][1]) + min_step))
    return winner_id, price


def _pick_next_player(available_pool: list[Player], order_seed: int) -> Player | None:
    if not available_pool:
        return None
    rng = np.random.default_rng(order_seed)
    indices = list(range(len(available_pool)))
    rng.shuffle(indices)
    return available_pool[indices[0]]


def _squad_full(participant: ParticipantState, role_quotas: Mapping[str, int]) -> bool:
    return _total_remaining_slots(_remaining_slots(participant, role_quotas)) <= 0


def _all_squads_full(state: AuctionState) -> bool:
    return all(
        _squad_full(p, state.config.role_quotas) for p in state.participants.values()
    )


def _run_one_scenario(
    participants,
    config,
    player_pool,
    profiles,
    scenario_idx,
    base_seed,
    price_noise_std_ratio,
    min_bid_step,
) -> AuctionState:
    state = initialize_auction(participants, config, player_pool)
    rng = np.random.default_rng(base_seed + scenario_idx * 1_000_003)
    order_seed = base_seed + scenario_idx * 97
    for round_i in range(len(player_pool) + 5):
        if _all_squads_full(state) or not state.available_pool:
            break
        player = _pick_next_player(state.available_pool, order_seed + round_i)
        if player is None:
            break
        bids: dict[str, float] = {}
        for pid, pstate in state.participants.items():
            policy = profiles.get(pid)
            if policy is None:
                continue
            bid = _bot_bid(player, pstate, state, policy, rng, price_noise_std_ratio)
            if bid is not None and bid >= 1:
                bids[pid] = bid
        if not bids:
            state.available_pool = [
                p for p in state.available_pool if p.player_id != player.player_id
            ]
            continue
        resolved = _resolve_winner(bids, min_step=min_bid_step)
        if resolved is None:
            continue
        winner_id, price = resolved
        price_int = max(1, round(price))
        winner = state.participants[winner_id]
        if price_int > winner.budget_residual:
            price_int = max(1, winner.budget_residual)
        result = record_assignment(
            state,
            player_id=player.player_id,
            winner_participant_id=winner_id,
            final_price=price_int,
        )
        if not result.success:
            state.available_pool = [
                p for p in state.available_pool if p.player_id != player.player_id
            ]
    return state


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _esv_proxy(player: Player, price: float) -> float:
    if player.esv is not None:
        return float(player.esv)
    score = float(player.projected_score or 0)
    return score if price <= 0 else score - price * 0.01


def simulate_auction(
    participants: list[ParticipantSetup],
    bidder_profiles: Sequence[BidderProfile],
    config: AuctionConfig,
    player_pool: list[Player],
    sim_config: AuctionSimulationConfig | None = None,
) -> AuctionSimulationResult:
    """Run n_simulations independent synthetic auctions. Stateless."""
    sim_config = sim_config or AuctionSimulationConfig()
    t0 = time.perf_counter()
    warnings: list[str] = []
    profiles = {bp.participant_id: bp.policy for bp in bidder_profiles}
    missing = [
        p.participant_id for p in participants if p.participant_id not in profiles
    ]
    if missing:
        default = BidderPolicy()
        for pid in missing:
            profiles[pid] = default
        warnings.append(
            f"Default BidderPolicy applied to participants without profile: {missing}"
        )

    spend_series: dict[str, list[float]] = defaultdict(list)
    esv_series: dict[str, list[float]] = defaultdict(list)
    completed_flags: dict[str, list[int]] = defaultdict(list)
    composition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    price_index_samples: list[dict[str, dict[str, float]]] = []
    acquisition: dict[str, list[tuple[str, float]]] = defaultdict(list)
    n_completed = 0

    for k in range(sim_config.n_simulations):
        if (
            sim_config.timeout_seconds > 0
            and (time.perf_counter() - t0) > sim_config.timeout_seconds
        ):
            warnings.append(
                f"Simulation stopped early at {k}/{sim_config.n_simulations} due to timeout"
            )
            break
        state = _run_one_scenario(
            participants,
            config,
            player_pool,
            profiles,
            k,
            sim_config.random_seed,
            sim_config.price_noise_std_ratio,
            sim_config.min_bid_step,
        )
        n_completed += 1
        price_index_samples.append(
            {
                str(role): {str(t): float(v) for t, v in tiers.items()}
                for role, tiers in state.price_index.items()
            }
        )
        for pid, pstate in state.participants.items():
            initial_budget = next(
                (p.budget_initial for p in participants if p.participant_id == pid),
                config.budget_initial,
            )
            spend_series[pid].append(float(initial_budget - pstate.budget_residual))
            esv_total = sum(
                _esv_proxy(a.player, float(a.final_price))
                for a in state.assignments
                if a.winner_participant_id == pid
            )
            esv_series[pid].append(esv_total)
            completed_flags[pid].append(
                1 if _squad_full(pstate, config.role_quotas) else 0
            )
            for role, count in pstate.role_breakdown.items():
                if count > 0:
                    composition_counts[pid][role] += count
            if pstate.budget_residual < 0:
                warnings.append(f"budget_residual < 0 for {pid} in scenario {k}")
        for a in state.assignments:
            acquisition[a.player.player_id].append(
                (a.winner_participant_id, float(a.final_price))
            )

    per_participant: dict[str, ParticipantSimStats] = {}
    for p in participants:
        pid = p.participant_id
        spends = spend_series.get(pid, [0.0])
        esvs = esv_series.get(pid, [0.0])
        flags = completed_flags.get(pid, [0])
        comp = composition_counts.get(pid, Counter())
        mode_comp = {
            role: round(total / max(n_completed, 1)) for role, total in comp.items()
        }
        per_participant[pid] = ParticipantSimStats(
            spend_p10=_percentile(spends, 10),
            spend_p50=_percentile(spends, 50),
            spend_p90=_percentile(spends, 90),
            esv_total_p10=_percentile(esvs, 10),
            esv_total_p50=_percentile(esvs, 50),
            esv_total_p90=_percentile(esvs, 90),
            completion_probability=float(np.mean(flags)) if flags else 0.0,
            squad_composition_mode=mode_comp,
        )

    price_index_drift_p50: dict[str, dict[str, float]] = {}
    if price_index_samples:
        for role in price_index_samples[0]:
            price_index_drift_p50[role] = {
                tier: _percentile(
                    [
                        s[role][tier]
                        for s in price_index_samples
                        if role in s and tier in s[role]
                    ],
                    50,
                )
                for tier in price_index_samples[0][role]
            }

    player_acq: dict[str, dict[str, float]] = {}
    for player_id, events in acquisition.items():
        if not events:
            continue
        prices = [price for _, price in events]
        player_acq[player_id] = {
            "prob": len(events) / max(n_completed, 1),
            "avg_price": float(np.mean(prices)),
        }

    return AuctionSimulationResult(
        n_completed=n_completed,
        per_participant=per_participant,
        price_index_drift_p50=price_index_drift_p50,
        player_acquisition_probability=player_acq,
        wall_time_seconds=time.perf_counter() - t0,
        warnings=warnings,
    )
