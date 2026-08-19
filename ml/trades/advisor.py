"""Trade advisor: coverage matrix, retention score, trade-out / trade-in lists.

Retention is **individual** (plan D9): team strength of the real Serie A club
never drives a sell recommendation.  Top scorers are hard-excluded above a
configurable threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ml.optimizer.formations import (
    MANTRA_FORMATIONS,
    FormationCoverage,
    MantraFormation,
    evaluate_coverage,
    get_formation,
)

log = logging.getLogger(__name__)

DEFAULT_HARD_EXCLUSION_THRESHOLD = 75.0
TOP5_BONUS = 15.0
TOP10_BONUS = 10.0


# ── Player view for the advisor ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TradePlayer:
    """Minimal player record for coverage / retention."""

    player_id: str
    name: str
    eligible_roles: frozenset[str]
    """Mantra roles — used by evaluate_coverage via duck-typing."""

    cost: int = 0
    current_value: int | None = None
    fp_corr: float = 50.0
    """MANTRA FP_Corr 0–100 (or hybrid equivalent)."""

    goals: int = 0
    assists: int = 0
    minutes: int = 0
    team_serie_a: str = ""
    is_top5_scorer_role: bool = False
    is_top10_scorer_role: bool = False

    @property
    def player_id_attr(self) -> str:
        return self.player_id


# evaluate_coverage expects objects with eligible_roles and optional player_id
def _coverage_player_view(p: TradePlayer) -> object:
    class _P:
        eligible_roles = p.eligible_roles
        player_id = p.player_id

    return _P()


# ── Retention score (D9) ─────────────────────────────────────────────────────


def retention_score(
    player: TradePlayer,
    *,
    top5_bonus: float = TOP5_BONUS,
    top10_bonus: float = TOP10_BONUS,
) -> float:
    """Individual hold score on a 0–100+ scale.

    Base = FP_Corr.  Bonuses for top scorers.  Team ranking is intentionally
    **not** used as a primary signal.
    """
    base = float(player.fp_corr)
    bonus = 0.0
    if player.is_top5_scorer_role:
        bonus += top5_bonus
    elif player.is_top10_scorer_role:
        bonus += top10_bonus
    # Light minutes reliability (optional, small)
    if player.minutes >= 1500:
        bonus += 3.0
    elif player.minutes > 0 and player.minutes < 500:
        bonus -= 5.0
    return base + bonus


# ── Coverage matrix ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CoverageCell:
    formation: str
    slot_label: str
    status: str  # "ok" | "deficit"
    missing: int = 0


def build_coverage_matrix(
    squad: Sequence[TradePlayer],
    formation_labels: Sequence[str],
) -> tuple[dict[str, FormationCoverage], list[CoverageCell]]:
    """Evaluate squad against each preferred formation."""
    views = [_coverage_player_view(p) for p in squad]
    coverages: dict[str, FormationCoverage] = {}
    cells: list[CoverageCell] = []

    for label in formation_labels:
        try:
            formation = get_formation(label)
        except KeyError:
            log.warning("Unknown formation %r — skipped", label)
            continue
        cov = evaluate_coverage(views, formation, require_por=True)
        coverages[label] = cov
        if cov.feasible:
            # mark all slot labels ok
            for slot in formation.slots:
                key = slot.label or "/".join(sorted(slot.roles))
                cells.append(
                    CoverageCell(formation=label, slot_label=key, status="ok")
                )
        else:
            for slot_label, missing in cov.deficits.items():
                cells.append(
                    CoverageCell(
                        formation=label,
                        slot_label=slot_label,
                        status="deficit",
                        missing=missing,
                    )
                )
            # also list non-deficit slots as ok for matrix completeness
            deficit_keys = set(cov.deficits)
            for slot in formation.slots:
                key = slot.label or "/".join(sorted(slot.roles))
                if key not in deficit_keys and key != "Por":
                    cells.append(
                        CoverageCell(formation=label, slot_label=key, status="ok")
                    )
    return coverages, cells


def _surplus_roles(
    squad: Sequence[TradePlayer],
    formation_labels: Sequence[str],
) -> set[str]:
    """Roles that appear more often than needed across the preferred modules."""
    demand: dict[str, int] = {}
    for label in formation_labels:
        try:
            formation = get_formation(label)
        except KeyError:
            continue
        for slot in formation.slots:
            for role in slot.roles:
                demand[role] = demand.get(role, 0) + slot.count
        # Por implicit
        demand["Por"] = demand.get("Por", 0) + 1

    supply: dict[str, int] = {}
    for p in squad:
        for r in p.eligible_roles:
            supply[r] = supply.get(r, 0) + 1

    surplus: set[str] = set()
    for role, sup in supply.items():
        # Heuristic: surplus if supply > max demand for that role across modules
        # (use per-module max rather than sum)
        max_demand = 0
        for label in formation_labels:
            try:
                formation = get_formation(label)
            except KeyError:
                continue
            d = 0
            for slot in formation.slots:
                if role in slot.roles:
                    d += slot.count
            if role == "Por":
                d = 1
            max_demand = max(max_demand, d)
        if max_demand > 0 and sup > max_demand:
            surplus.add(role)
        elif max_demand == 0 and sup > 0:
            # role never required — soft surplus
            surplus.add(role)
    return surplus


# ── Trade-out / trade-in ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TradeOutCandidate:
    player: TradePlayer
    retention: float
    surplus_roles: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class TradeInTarget:
    player_id: str
    name: str
    covers_slots: tuple[str, ...]
    fp_corr: float
    estimated_cost: int
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExcludedTopPerformer:
    player: TradePlayer
    retention: float
    reason: str


@dataclass(frozen=True, slots=True)
class TradeDashboard:
    formation_prefs: tuple[str, ...]
    coverage_cells: tuple[CoverageCell, ...]
    coverage_by_formation: dict[str, bool]
    """formation label → feasible."""

    trade_out: tuple[TradeOutCandidate, ...]
    trade_in: tuple[TradeInTarget, ...]
    excluded_top_performers: tuple[ExcludedTopPerformer, ...]


def rank_trade_out_candidates(
    squad: Sequence[TradePlayer],
    formation_labels: Sequence[str],
    *,
    hard_exclusion_threshold: float = DEFAULT_HARD_EXCLUSION_THRESHOLD,
) -> tuple[list[TradeOutCandidate], list[ExcludedTopPerformer]]:
    surplus = _surplus_roles(squad, formation_labels)
    outs: list[TradeOutCandidate] = []
    excluded: list[ExcludedTopPerformer] = []

    for p in squad:
        player_surplus = tuple(sorted(r for r in p.eligible_roles if r in surplus))
        if not player_surplus:
            continue

        ret = retention_score(p)
        if ret >= hard_exclusion_threshold:
            reason = (
                f"retention={ret:.1f} ≥ {hard_exclusion_threshold} "
                f"(FP_Corr={p.fp_corr}"
            )
            if p.is_top5_scorer_role:
                reason += ", Top-5 marcatori ruolo"
            elif p.is_top10_scorer_role:
                reason += ", Top-10 marcatori ruolo"
            reason += ") — escluso da cessione automatica"
            excluded.append(
                ExcludedTopPerformer(player=p, retention=ret, reason=reason)
            )
            continue

        rationale = (
            f"Ruoli in surplus: {', '.join(player_surplus)}; "
            f"retention={ret:.1f} (FP_Corr={p.fp_corr})"
        )
        outs.append(
            TradeOutCandidate(
                player=p,
                retention=ret,
                surplus_roles=player_surplus,
                rationale=rationale,
            )
        )

    outs.sort(key=lambda c: c.retention)  # lowest retention first
    return outs, excluded


def rank_trade_in_targets(
    *,
    deficit_slot_roles: Sequence[frozenset[str] | set[str]],
    market_pool: Sequence[TradePlayer],
    budget: int | None = None,
    limit: int = 15,
) -> list[TradeInTarget]:
    """Rank free-market (or external) players that cover deficit OR-groups.

    ``market_pool`` is supplied by the caller (svincolati / other teams' surplus
    in v2).  Polivalenza: players covering more deficit groups rank higher.
    """
    if not deficit_slot_roles:
        return []

    targets: list[TradeInTarget] = []
    for p in market_pool:
        covers: list[str] = []
        for roles in deficit_slot_roles:
            roles_fs = frozenset(roles)
            if p.eligible_roles & roles_fs:
                covers.append("/".join(sorted(roles_fs)))
        if not covers:
            continue
        cost = p.current_value if p.current_value is not None else p.cost
        if budget is not None and cost > budget:
            continue
        targets.append(
            TradeInTarget(
                player_id=p.player_id,
                name=p.name,
                covers_slots=tuple(covers),
                fp_corr=p.fp_corr,
                estimated_cost=cost,
                roles=tuple(sorted(p.eligible_roles)),
            )
        )

    targets.sort(
        key=lambda t: (len(t.covers_slots), t.fp_corr),
        reverse=True,
    )
    return targets[:limit]


def build_trade_dashboard(
    squad: Sequence[TradePlayer],
    formation_prefs: Sequence[str],
    *,
    market_pool: Sequence[TradePlayer] = (),
    hard_exclusion_threshold: float = DEFAULT_HARD_EXCLUSION_THRESHOLD,
    budget: int | None = None,
) -> TradeDashboard:
    """Full dashboard payload for the Scambi tab."""
    if not formation_prefs:
        formation_prefs = ["4-3-3", "3-5-2", "3-4-3"]

    coverages, cells = build_coverage_matrix(squad, formation_prefs)
    coverage_by_formation = {
        label: cov.feasible for label, cov in coverages.items()
    }

    outs, excluded = rank_trade_out_candidates(
        squad,
        formation_prefs,
        hard_exclusion_threshold=hard_exclusion_threshold,
    )

    # Collect deficit role-groups across prefs
    deficit_groups: list[frozenset[str]] = []
    for label, cov in coverages.items():
        if cov.feasible:
            continue
        try:
            formation = get_formation(label)
        except KeyError:
            continue
        for slot in formation.slots:
            key = slot.label or "/".join(sorted(slot.roles))
            if key in cov.deficits or "Por" in cov.deficits and key == "Por":
                deficit_groups.append(frozenset(slot.roles))
        if "Por" in cov.deficits:
            deficit_groups.append(frozenset({"Por"}))

    # Unique deficit groups
    unique_deficits: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for g in deficit_groups:
        if g not in seen:
            seen.add(g)
            unique_deficits.append(g)

    trade_in = rank_trade_in_targets(
        deficit_slot_roles=unique_deficits,
        market_pool=market_pool,
        budget=budget,
    )

    return TradeDashboard(
        formation_prefs=tuple(formation_prefs),
        coverage_cells=tuple(cells),
        coverage_by_formation=coverage_by_formation,
        trade_out=tuple(outs),
        trade_in=tuple(trade_in),
        excluded_top_performers=tuple(excluded),
    )
