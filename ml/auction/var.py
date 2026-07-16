"""Auction Value Above Replacement (VAR) and Expected Surplus Value.

VAR measures how much better a player is than the best freely available
replacement at the same role. It drives Expected Surplus Value (ESV):

    ESV = (VAR / baseline_var) * budget_per_slot - expected_price

demand_curve is parametric (monotone convex in VAR) and is explicitly
marked calibrated=False until real auction history is provided.

Integration boundary
--------------------
`expected_price` from this module feeds `price_drift.compute_baseline_cost`
as `baseline_cost` input. It does NOT replace the EWMA price-drift model.
See ml/auction/__init__.py for the full boundary contract.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── ReplacementLevel ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplacementLevel:
    """Replacement-level score per role.

    Built from the bottom N players at each role who are expected to be
    available undrafted (or at minimum bid). Default: bottom 10% by projected
    score within each role pool.

    Args:
        role: Fantacalcio role code.
        score: Replacement-level projected fantavoto.
        n_players_used: Number of players used to compute the median.
        percentile_threshold: Fraction of players considered replaceable.
    """

    role: str
    score: float
    n_players_used: int
    percentile_threshold: float = 0.10  # bottom 10%

    @classmethod
    def from_player_pool(
        cls,
        role: str,
        scores: list[float],
        percentile_threshold: float = 0.10,
    ) -> "ReplacementLevel":
        """Compute replacement level from the bottom percentile of a role pool.

        Args:
            role: Role code.
            scores: All projected scores for this role.
            percentile_threshold: Bottom fraction considered replaceable.

        Returns:
            ReplacementLevel with median score of the bottom tier.

        Raises:
            ValueError: If scores is empty.
        """
        if not scores:
            raise ValueError(f"Cannot compute ReplacementLevel for role '{role}': empty scores.")
        import numpy as np

        sorted_scores = sorted(scores)
        n = max(1, int(math.ceil(len(sorted_scores) * percentile_threshold)))
        replacement_score = float(np.median(sorted_scores[:n]))
        return cls(
            role=role,
            score=replacement_score,
            n_players_used=n,
            percentile_threshold=percentile_threshold,
        )


# ── VAR ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VAR:
    """Value Above Replacement for a single player.

    var_score = projected_score - replacement_level.score
    Negative VAR means the player is below replacement level.
    """

    player_id: str
    role: str
    projected_score: float
    replacement_level_score: float
    var_score: float  # = projected_score - replacement_level_score

    @classmethod
    def compute(
        cls,
        player_id: str,
        role: str,
        projected_score: float,
        replacement_level: ReplacementLevel,
    ) -> "VAR":
        """Compute VAR for a player.

        Args:
            player_id: Unique player identifier.
            role: Fantacalcio role code.
            projected_score: Ensemble point prediction.
            replacement_level: Role's replacement-level baseline.

        Returns:
            VAR with var_score = projected_score - replacement_level.score.
        """
        return cls(
            player_id=player_id,
            role=role,
            projected_score=projected_score,
            replacement_level_score=replacement_level.score,
            var_score=projected_score - replacement_level.score,
        )


# ── DemandCurve ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DemandCurve:
    """Parametric demand curve: expected auction price as a function of VAR.

    price(var) = base_price + scale * max(0, var) ^ exponent

    This is a monotone convex function in VAR. The parameters are initial
    priors — NOT calibrated against real auction data.

    calibrated: False until real auction history is fitted.
    When calibrated=False, a warning is logged on every call to expected_price().
    """

    base_price: float = 1.0     # minimum bid (Fantacalcio: 1 credit)
    scale: float = 8.0          # price sensitivity to VAR
    exponent: float = 1.4       # convexity (>1 = convex, top players get premium)
    calibrated: bool = False    # MUST be False until fitted on real auction data

    def expected_price(self, var_score: float) -> float:
        """Compute expected auction price for a given VAR.

        Args:
            var_score: Value Above Replacement score.

        Returns:
            Expected auction price in credits. Minimum is base_price.
        """
        if not self.calibrated:
            log.warning(
                "DemandCurve is not calibrated (calibrated=False). "
                "expected_price() returns parametric prior, not empirical estimate. "
                "Fit on real auction history before using for bid decisions."
            )
        positive_var = max(0.0, var_score)
        price = self.base_price + self.scale * (positive_var**self.exponent)
        return round(price, 2)


# ── ExpectedSurplusValue ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpectedSurplusValue:
    """Expected Surplus Value: how much a player is worth over their expected price.

    ESV > 0: player is expected to be underpriced → buy signal.
    ESV < 0: player is expected to be overpriced → avoid signal.

    budget_per_slot = total_budget / n_roster_slots (for this role).
    """

    player_id: str
    role: str
    var_score: float
    expected_price: float
    budget_per_slot: float
    esv: float   # = expected_performance_value - expected_price
    calibrated: bool  # mirrors DemandCurve.calibrated

    @classmethod
    def compute(
        cls,
        var: VAR,
        demand_curve: DemandCurve,
        budget_per_slot: float,
        baseline_var: float,
    ) -> "ExpectedSurplusValue":
        """Compute ESV for a player.

        Args:
            var: Computed VAR for this player.
            demand_curve: Parametric or fitted demand curve.
            budget_per_slot: Credits allocated per roster slot for this role.
            baseline_var: Normalisation factor (e.g. mean positive VAR in role).

        Returns:
            ExpectedSurplusValue with esv = perf_value - expected_price.
        """
        expected_px = demand_curve.expected_price(var.var_score)

        # Performance value: fraction of budget proportional to VAR above replacement
        # Uses baseline_var to normalise; clamped at 0 for negative VAR players
        if baseline_var > 0 and var.var_score > 0:
            perf_value = (var.var_score / baseline_var) * budget_per_slot
        else:
            perf_value = demand_curve.base_price  # replacement-level player = min bid value

        esv = perf_value - expected_px

        return cls(
            player_id=var.player_id,
            role=var.role,
            var_score=var.var_score,
            expected_price=expected_px,
            budget_per_slot=budget_per_slot,
            esv=esv,
            calibrated=demand_curve.calibrated,
        )


# ── VarEngine ────────────────────────────────────────────────────────────────


class VarEngine:
    """Orchestrates VAR + ESV computation for a full player pool.

    Usage::

        engine = VarEngine(demand_curve=DemandCurve(), total_budget=500)
        results = engine.evaluate(players)  # players: list[dict]

    Each dict in players: {player_id, role, projected_score}.
    Returns list[ExpectedSurplusValue], sorted by ESV descending.
    """

    def __init__(
        self,
        demand_curve: DemandCurve | None = None,
        total_budget: int = 500,
        roster_slots: dict[str, int] | None = None,
        percentile_threshold: float = 0.10,
    ) -> None:
        self.demand_curve = demand_curve or DemandCurve()
        self.total_budget = total_budget
        # Default Fantacalcio classico slots: P=3, D=8, C=8, A=6
        self.roster_slots = roster_slots or {"P": 3, "D": 8, "C": 8, "A": 6}
        self.percentile_threshold = percentile_threshold

    def evaluate(self, players: list[dict]) -> list[ExpectedSurplusValue]:
        """Compute VAR and ESV for all players.

        Args:
            players: List of dicts with player_id, role, projected_score.

        Returns:
            List of ExpectedSurplusValue sorted by ESV descending.
        """
        by_role: dict[str, list[dict]] = defaultdict(list)
        for p in players:
            by_role[str(p["role"])].append(p)

        total_slots = sum(self.roster_slots.values())
        # ponytail: uniform budget_per_slot = total/total_slots; role-weighted if needed
        budget_per_slot = self.total_budget / total_slots if total_slots > 0 else self.total_budget

        results: list[ExpectedSurplusValue] = []

        for role, role_players in by_role.items():
            scores = [float(p["projected_score"]) for p in role_players]
            replacement = ReplacementLevel.from_player_pool(
                role, scores, self.percentile_threshold
            )

            vars_ = [
                VAR.compute(
                    str(p["player_id"]), role, float(p["projected_score"]), replacement
                )
                for p in role_players
            ]

            positive_vars = [v.var_score for v in vars_ if v.var_score > 0]
            baseline_var = sum(positive_vars) / len(positive_vars) if positive_vars else 1.0

            for v in vars_:
                esv = ExpectedSurplusValue.compute(
                    v, self.demand_curve, budget_per_slot, baseline_var
                )
                results.append(esv)

        return sorted(results, key=lambda e: e.esv, reverse=True)
