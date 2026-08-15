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

Multi-role (MANTRA) replacement-level policy
---------------------------------------------
A CLASSIC player has exactly one role, so its ReplacementLevel is
unambiguous: the one for ``player["role"]``. A MANTRA player may instead be
eligible for several roles (``player["eligible_roles"]``) — e.g. a
wing-back eligible for both ``Dd`` and ``E``. Two things need to be
decided for such a player, and this module makes both decisions
explicitly rather than leaving them as an accident of implementation:

1. **Replacement-level pool membership**: a multi-role player is counted
   as available supply for *every* role they are eligible for, not just
   one — this is a supply-side fact (they could end up filling any of
   those slots) and affects how "deep" each of those roles' pools looks.
2. **Which single ReplacementLevel values their own VAR**: the player is
   only ever going to fill *one* slot, so VAR needs one baseline. Policy
   (per audit recommendation): use the ReplacementLevel of the
   **scarcest** eligible role, operationalised here as the one with the
   *highest* replacement-level score. A high replacement-level score
   means even bottom-of-pool players at that role are still valuable —
   i.e. the role's depth is thin. Anchoring VAR to the scarcest role is
   the conservative choice: it prevents a flex player's surplus value
   from being inflated by picking whichever eligible role happens to
   have the shallowest bottom (which would overstate how much better
   than "replaceable" they really are). See :func:`_select_scarcest_replacement`.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from ml.auction.models import ValuationMode

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

    @classmethod
    def from_roster_depth(
        cls,
        role: str,
        scores: list[float],
        num_participants: int,
        role_quota: int,
    ) -> "ReplacementLevel":
        """Replacement level = score at position (num_participants × role_quota) in descending sort.

        This models the last player drafted at a role in a league of N teams,
        each filling `role_quota` slots. Players below this rank are undrafted.

        Args:
            role: Role code.
            scores: All projected scores for this role.
            num_participants: Number of teams in the league.
            role_quota: Number of roster slots per team for this role.

        Returns:
            ReplacementLevel with the score at the draft cutoff.

        Raises:
            ValueError: If scores is empty.
        """
        if not scores:
            raise ValueError(f"Cannot compute ReplacementLevel for role '{role}': empty scores.")
        sorted_desc = sorted(scores, reverse=True)
        cutoff_idx = min(num_participants * role_quota, len(sorted_desc)) - 1
        cutoff_idx = max(0, cutoff_idx)
        return cls(
            role=role,
            score=sorted_desc[cutoff_idx],
            n_players_used=cutoff_idx + 1,
            percentile_threshold=0.0,
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
        override_expected_price: float | None = None,
    ) -> "ExpectedSurplusValue":
        """Compute ESV for a player.

        Args:
            var: Computed VAR for this player.
            demand_curve: Parametric or fitted demand curve.
            budget_per_slot: Credits allocated per roster slot for this role.
            baseline_var: Normalisation factor (e.g. mean positive VAR in role).
            override_expected_price: If provided, use this price instead of
                demand_curve.expected_price(). Pass the EWMA projection from
                price_drift when inside a live session.

        Returns:
            ExpectedSurplusValue with esv = perf_value - expected_price.
        """
        expected_px = override_expected_price if override_expected_price is not None \
            else demand_curve.expected_price(var.var_score)

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


def _select_scarcest_replacement(
    eligible_roles: frozenset[str] | set[str],
    replacement_by_role: dict[str, "ReplacementLevel"],
) -> "ReplacementLevel":
    """Pick the ReplacementLevel to VAR a multi-role player against.

    Policy: the eligible role whose ReplacementLevel.score is *highest*
    (the scarcest — see module docstring for the rationale). Only roles
    that actually have a computed ReplacementLevel in this pool are
    considered (a player's ``eligible_roles`` may list a role with no
    players at all in the current pool, e.g. a small MANTRA sub-pool).

    Ties broken by role code for determinism.

    Raises:
        ValueError: If none of ``eligible_roles`` has a ReplacementLevel
            in ``replacement_by_role`` (caller error: player pool and
            eligible_roles are inconsistent).
    """
    candidates = [
        replacement_by_role[r] for r in eligible_roles if r in replacement_by_role
    ]
    if not candidates:
        raise ValueError(
            f"None of eligible_roles {sorted(eligible_roles)} has a "
            "ReplacementLevel in the current pool"
        )
    return max(candidates, key=lambda rl: (rl.score, rl.role))


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
        valuation_mode: ValuationMode = ValuationMode.PER_MATCH_RATING,
        replacement_method: str = "percentile",
        num_participants: int = 8,
        min_start_probability: float | None = None,
        hybrid_blend: float = 0.0,
        risk_aversion: float = 0.0,
        apply_reliability_weight: bool = True,
    ) -> None:
        self.demand_curve = demand_curve or DemandCurve()
        self.total_budget = total_budget
        # Default Fantacalcio classico slots: P=3, D=8, C=8, A=6
        self.roster_slots = roster_slots or {"P": 3, "D": 8, "C": 8, "A": 6}
        self.percentile_threshold = percentile_threshold
        self.valuation_mode = valuation_mode
        self.replacement_method = replacement_method  # "percentile" | "roster_depth"
        self.num_participants = num_participants
        self.min_start_probability = min_start_probability
        # WS3 #2: convex blend with fpIbrido (MANTRA-ibrido signal), same
        # shape as OptimizationConfig.hybrid_blend. 0.0 = disabled (default).
        if not 0.0 <= hybrid_blend <= 1.0:
            raise ValueError(f"hybrid_blend must be in [0, 1], got {hybrid_blend}")
        self.hybrid_blend = hybrid_blend
        # plan-limited-cohort-hardening WS3: optional risk penalty and
        # reliability_weight multiplier, symmetric with Optimizer solver.
        # risk_aversion default 0.0 remains opt-in; apply_reliability_weight
        # defaults to True after ADR 0001 (align with Optimizer).
        if risk_aversion < 0.0:
            raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
        self.risk_aversion = float(risk_aversion)
        self.apply_reliability_weight = bool(apply_reliability_weight)

    def _pool_roles(self, player: dict) -> list[str]:
        """Roles this player counts as supply for.

        CLASSIC / single-role MANTRA: ``[player["role"]]`` — unchanged
        behaviour. Multi-role MANTRA: every role in ``eligible_roles``
        (sorted for deterministic downstream tie-breaks).
        """
        eligible = player.get("eligible_roles")
        if eligible:
            roles = sorted(str(r) for r in eligible)
            if roles:
                return roles
        return [str(player["role"])]

    def _get_score(self, player: dict) -> float:
        """Extract the relevant score based on valuation_mode.

        When ``hybrid_blend > 0`` and the player carries ``fp_ibrido``
        (voto-scale signal from :mod:`ml.optimizer.hybrid_loader`), the
        base score is blended as
        ``(1 - hybrid_blend) * base + hybrid_blend * fp_ibrido`` —
        same convex combination used by the optimizer objective.
        Players without ``fp_ibrido`` keep the pure base score.
        """
        if self.valuation_mode == ValuationMode.SEASON_VALUE:
            sv = player.get("season_value")
            if isinstance(sv, (int, float)) and sv > 0:
                base = float(sv)
            else:
                log.warning(
                    "Player %s missing season_value in SEASON_VALUE mode, "
                    "falling back to projected_score",
                    player.get("player_id"),
                )
                base = float(player["projected_score"])
        else:
            base = float(player["projected_score"])

        if self.hybrid_blend > 0.0:
            fp = player.get("fp_ibrido")
            if isinstance(fp, (int, float)) and fp > 0:
                base = (1.0 - self.hybrid_blend) * base + self.hybrid_blend * float(fp)

        # Canonical decision policy (WS6 / ADR 0001). Hybrid blend is applied
        # first (auction-specific); reliability + risk then via shared helper.
        from ml.auction.decision_score import compute_decision_score
        return compute_decision_score(
            projected_score=base,
            reliability_weight=player.get("reliability_weight"),
            prediction_std=player.get("prediction_std"),
            apply_reliability_weight=self.apply_reliability_weight,
            risk_aversion=self.risk_aversion,
        )

    def evaluate(
        self,
        players: list[dict],
        price_overrides: dict[str, float] | None = None,
    ) -> list[ExpectedSurplusValue]:
        """Compute VAR and ESV for all players.

        Args:
            players: List of dicts with player_id, role, projected_score
                (and optionally season_value for SEASON_VALUE mode).
                MANTRA-only: a player dict may instead (or in addition)
                carry ``eligible_roles`` (iterable of role codes). When
                present with more than one role, the player is counted as
                supply for every eligible role's pool, and their own VAR
                is computed against the scarcest of those roles — see the
                module docstring for the exact policy. Absent or
                single-role ``eligible_roles`` behaves identically to
                CLASSIC (backward compatible).
            price_overrides: Optional map of player_id -> expected_price.
                When provided (e.g. from EWMA price_drift in a live session),
                bypasses DemandCurve for those players.

        Returns:
            List of ExpectedSurplusValue sorted by ESV descending.
        """
        by_role: dict[str, list[dict]] = defaultdict(list)
        for p in players:
            for r in self._pool_roles(p):
                by_role[r].append(p)

        total_slots = sum(self.roster_slots.values())
        # ponytail: uniform budget_per_slot = total/total_slots; role-weighted if needed
        budget_per_slot = self.total_budget / total_slots if total_slots > 0 else self.total_budget

        replacement_by_role: dict[str, ReplacementLevel] = {}
        for role, role_players in by_role.items():
            scores = [self._get_score(p) for p in role_players]
            if self.replacement_method == "roster_depth":
                role_quota = self.roster_slots.get(role, 6)
                replacement_by_role[role] = ReplacementLevel.from_roster_depth(
                    role, scores, self.num_participants, role_quota
                )
            else:
                replacement_by_role[role] = ReplacementLevel.from_player_pool(
                    role, scores, self.percentile_threshold
                )

        # Second pass: one VAR per player (not per role-pool membership),
        # against the single ReplacementLevel the policy selects.
        vars_by_role: dict[str, list[VAR]] = defaultdict(list)
        for p in players:
            eligible = self._pool_roles(p)
            replacement = (
                replacement_by_role[eligible[0]]
                if len(eligible) == 1
                else _select_scarcest_replacement(
                    frozenset(eligible), replacement_by_role
                )
            )
            v = VAR.compute(
                str(p["player_id"]), replacement.role, self._get_score(p), replacement
            )
            vars_by_role[replacement.role].append(v)

        results: list[ExpectedSurplusValue] = []
        overrides = price_overrides or {}

        for role, vars_ in vars_by_role.items():
            positive_vars = [v.var_score for v in vars_ if v.var_score > 0]
            baseline_var = sum(positive_vars) / len(positive_vars) if positive_vars else 1.0

            for v in vars_:
                esv = ExpectedSurplusValue.compute(
                    v,
                    self.demand_curve,
                    budget_per_slot,
                    baseline_var,
                    override_expected_price=overrides.get(v.player_id),
                )
                results.append(esv)

        ranked = sorted(results, key=lambda e: e.esv, reverse=True)

        if self.min_start_probability is not None:
            sp_map = {
                str(p["player_id"]): p.get("start_probability")
                for role_players in by_role.values()
                for p in role_players
            }
            ranked = [
                r for r in ranked
                if (sp_map.get(r.player_id) or 0.0) >= self.min_start_probability
            ]

        return ranked
