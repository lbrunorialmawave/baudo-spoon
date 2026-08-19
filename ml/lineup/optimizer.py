"""Exact single-matchday lineup optimizer.

For each official Mantra formation (or a caller-supplied subset) we solve a
maximum-weight assignment of eligible players to expanded slots via the
Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).  The module
with the highest total EV among feasible solutions is returned.

This is exact (not greedy): with ≤25 players and ≤11 outfield slots the
problem is tiny.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.optimizer.formations import (
    MANTRA_FORMATIONS,
    MantraFormation,
    SlotRequirement,
    get_formation,
)

log = logging.getLogger(__name__)

DEFAULT_MIN_STARTER_PROB = 0.15
_INFEASIBLE = -1e9


# ── Input / output types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LineupCandidate:
    """One owned player with pre-computed matchday expected value."""

    player_id: str
    """Stable id (fantacalcio_id as str, or name_clean fallback)."""

    name: str
    eligible_roles: frozenset[str]
    """Mantra role codes this player can fill."""

    expected_value: float
    """EV_giornata already including starter prob × opponent adjustment."""

    starter_probability: float = 1.0
    cost: int = 0
    team_serie_a: str = ""
    breakdown_note: str = ""


@dataclass(frozen=True, slots=True)
class SlotAssignment:
    slot_label: str
    slot_roles: frozenset[str]
    player_id: str
    player_name: str
    expected_value: float
    starter_probability: float
    breakdown_note: str = ""


@dataclass(frozen=True, slots=True)
class FormationResult:
    formation: str
    feasible: bool
    score_totale: float = 0.0
    assignments: tuple[SlotAssignment, ...] = ()
    reason: str = ""
    """Filled when infeasible (e.g. missing roles)."""

    gk: SlotAssignment | None = None
    """Chosen goalkeeper (always one Por when feasible)."""


@dataclass(frozen=True, slots=True)
class OptimizeResult:
    chosen: FormationResult | None
    alternatives: tuple[FormationResult, ...]
    bench: tuple[LineupCandidate, ...]
    """Remaining players ordered by EV desc (for manual swaps)."""


# ── Core algorithm ───────────────────────────────────────────────────────────


def _expand_slots(formation: MantraFormation) -> list[tuple[str, frozenset[str]]]:
    """Expand SlotRequirement.count into individual (label, roles) instances."""
    out: list[tuple[str, frozenset[str]]] = []
    for slot in formation.slots:
        label = slot.label or "/".join(sorted(slot.roles))
        for i in range(slot.count):
            # Distinguish multi-count slots for clarity in UI
            inst_label = label if slot.count == 1 else f"{label}#{i+1}"
            out.append((inst_label, slot.roles))
    return out


def _pick_goalkeeper(
    candidates: Sequence[LineupCandidate],
    *,
    min_starter_prob: float,
) -> LineupCandidate | None:
    por = [
        c
        for c in candidates
        if "Por" in c.eligible_roles and c.starter_probability >= min_starter_prob
    ]
    if not por:
        # fallback: any Por regardless of probability
        por = [c for c in candidates if "Por" in c.eligible_roles]
    if not por:
        return None
    return max(por, key=lambda c: c.expected_value)


def _assign_formation(
    formation: MantraFormation,
    outfield: Sequence[LineupCandidate],
    *,
    min_starter_prob: float,
) -> FormationResult:
    """Solve max-weight assignment for one formation's outfield slots."""
    slots = _expand_slots(formation)
    n_slots = len(slots)
    if n_slots == 0:
        return FormationResult(
            formation=formation.label,
            feasible=False,
            reason="formation has no outfield slots",
        )

    eligible = [
        c
        for c in outfield
        if c.starter_probability >= min_starter_prob and c.eligible_roles
    ]
    if len(eligible) < n_slots:
        return FormationResult(
            formation=formation.label,
            feasible=False,
            reason=f"only {len(eligible)} eligible outfield players for {n_slots} slots",
        )

    n_players = len(eligible)
    # Cost matrix: rows = players, cols = slots.  We maximise → use negative costs.
    # Pad to square.
    n = max(n_players, n_slots)
    cost = np.full((n, n), -_INFEASIBLE, dtype=float)  # large positive = bad when minimising

    for i, player in enumerate(eligible):
        for j, (_label, roles) in enumerate(slots):
            if player.eligible_roles & roles:
                # minimise → negate EV
                cost[i, j] = -player.expected_value
            # else leave as infeasible (large positive)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: list[SlotAssignment] = []
    total = 0.0
    used_player_ids: set[str] = set()
    missing: list[str] = []

    for r, c in zip(row_ind, col_ind):
        if c >= n_slots or r >= n_players:
            continue
        if cost[r, c] >= -_INFEASIBLE / 2:  # infeasible edge selected
            missing.append(slots[c][0])
            continue
        player = eligible[r]
        label, roles = slots[c]
        ev = player.expected_value
        total += ev
        used_player_ids.add(player.player_id)
        assignments.append(
            SlotAssignment(
                slot_label=label,
                slot_roles=roles,
                player_id=player.player_id,
                player_name=player.name,
                expected_value=round(ev, 4),
                starter_probability=player.starter_probability,
                breakdown_note=player.breakdown_note,
            )
        )

    if missing or len(assignments) < n_slots:
        reason_parts = []
        if missing:
            reason_parts.append(f"slot(s) uncovered: {', '.join(missing)}")
        if len(assignments) < n_slots:
            reason_parts.append(
                f"assigned {len(assignments)}/{n_slots} slots"
            )
        return FormationResult(
            formation=formation.label,
            feasible=False,
            score_totale=round(total, 4),
            assignments=tuple(assignments),
            reason="; ".join(reason_parts) or "infeasible assignment",
        )

    return FormationResult(
        formation=formation.label,
        feasible=True,
        score_totale=round(total, 4),
        assignments=tuple(assignments),
    )


def optimize_lineup(
    candidates: Sequence[LineupCandidate],
    *,
    formations: Sequence[str] | Sequence[MantraFormation] | None = None,
    min_starter_prob: float = DEFAULT_MIN_STARTER_PROB,
) -> OptimizeResult:
    """Pick the best feasible formation + assignment for the given squad.

    Parameters
    ----------
    candidates:
        Owned players with roles and pre-computed EV.
    formations:
        Subset of module labels or ``MantraFormation`` objects.  Default: full
        official Mantra catalog.
    min_starter_prob:
        Players below this starter probability are excluded from starting XI
        (plan D8).
    """
    if not candidates:
        return OptimizeResult(chosen=None, alternatives=(), bench=())

    # Resolve formation catalog
    resolved: list[MantraFormation] = []
    if formations is None:
        resolved = list(MANTRA_FORMATIONS)
    else:
        for f in formations:
            if isinstance(f, MantraFormation):
                resolved.append(f)
            else:
                resolved.append(get_formation(str(f)))

    gk = _pick_goalkeeper(candidates, min_starter_prob=min_starter_prob)
    # Pure Por stay out of the outfield pool; multi-role players remain eligible.
    outfield = [
        c for c in candidates if not (c.eligible_roles <= frozenset({"Por"}))
    ]

    results: list[FormationResult] = []
    for formation in resolved:
        fr = _assign_formation(
            formation, outfield, min_starter_prob=min_starter_prob
        )
        if fr.feasible and gk is not None:
            gk_slot = SlotAssignment(
                slot_label="Por",
                slot_roles=frozenset({"Por"}),
                player_id=gk.player_id,
                player_name=gk.name,
                expected_value=round(gk.expected_value, 4),
                starter_probability=gk.starter_probability,
                breakdown_note=gk.breakdown_note,
            )
            fr = FormationResult(
                formation=fr.formation,
                feasible=True,
                score_totale=round(fr.score_totale + gk.expected_value, 4),
                assignments=fr.assignments,
                gk=gk_slot,
            )
        elif fr.feasible and gk is None:
            fr = FormationResult(
                formation=fr.formation,
                feasible=False,
                score_totale=fr.score_totale,
                assignments=fr.assignments,
                reason="no eligible goalkeeper (Por)",
            )
        results.append(fr)

    feasible = [r for r in results if r.feasible]
    if not feasible:
        return OptimizeResult(
            chosen=None,
            alternatives=tuple(
                sorted(results, key=lambda r: r.score_totale, reverse=True)
            ),
            bench=tuple(
                sorted(candidates, key=lambda c: c.expected_value, reverse=True)
            ),
        )

    # Tie-break: higher score, then higher mean starter_probability of XI
    def _sort_key(r: FormationResult) -> tuple[float, float]:
        mean_sp = 0.0
        n = 0
        if r.gk:
            mean_sp += r.gk.starter_probability
            n += 1
        for a in r.assignments:
            mean_sp += a.starter_probability
            n += 1
        mean_sp = mean_sp / n if n else 0.0
        return (r.score_totale, mean_sp)

    feasible.sort(key=_sort_key, reverse=True)
    chosen = feasible[0]

    # Bench = not in chosen XI
    used = {a.player_id for a in chosen.assignments}
    if chosen.gk:
        used.add(chosen.gk.player_id)
    bench = tuple(
        sorted(
            (c for c in candidates if c.player_id not in used),
            key=lambda c: c.expected_value,
            reverse=True,
        )
    )

    # Alternatives: all other evaluated (feasible first)
    alts = tuple(
        sorted(
            (r for r in results if r.formation != chosen.formation),
            key=lambda r: (r.feasible, r.score_totale),
            reverse=True,
        )
    )

    return OptimizeResult(chosen=chosen, alternatives=alts, bench=bench)


# ── EV helper (optional, used by API layer) ──────────────────────────────────


def compute_ev(
    *,
    fp_ibrido_voto: float,
    starter_probability: float,
    opponent_adjustment: float = 1.0,
) -> float:
    """EV_giornata = FP_Ibrido_voto × StarterProbability × OpponentAdjustment."""
    return float(fp_ibrido_voto) * float(starter_probability) * float(opponent_adjustment)


def opponent_adjustment(
    role: str,
    opponent_strength_effective: float,
    *,
    k_att: float = 0.30,
    k_def: float = 0.20,
) -> float:
    """Asymmetric opponent adjustment (plan D7).

    ``opponent_strength_effective`` ∈ [0, 1]; higher = stronger opponent.
    """
    delta = 0.5 - float(opponent_strength_effective)
    offensive = {"M", "C", "T", "W", "A", "Pc"}
    defensive = {"Por", "Dc", "Dd", "Ds", "B", "E"}
    if role in offensive:
        adj = 1.0 + delta * k_att
        return float(max(0.85, min(1.15, adj)))
    if role in defensive:
        adj = 1.0 + delta * k_def
        return float(max(0.90, min(1.10, adj)))
    return 1.0
