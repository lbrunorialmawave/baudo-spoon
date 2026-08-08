"""Mantra Experience formation catalog and squad-level coverage evaluation.

Single source of truth for the 11 official Mantra modules (2026/27).
Used by Optimizer (post-hoc + optional hard constraint) and Auction
(residual coverage). Pure domain logic — no solver / API imports.

Phase 0
-------
OR-groups and module slots are frozen in
``docs/mantra_formations_2026_27.md`` (catalog version
``MANTRA_FORMATIONS_V2026_27``). Update that doc and this module together
if the official image changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ml.mantra.roles import ALL_ROLES

__all__ = [
    "SlotRequirement",
    "MantraFormation",
    "FormationCoverage",
    "MANTRA_FORMATIONS",
    "MANTRA_FORMATIONS_BY_LABEL",
    "evaluate_coverage",
    "evaluate_all_coverages",
    "get_formation",
]


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotRequirement:
    """One tactical line requirement that can be filled by alternative roles.

    A player may fill at most one slot.  ``roles`` is an OR-group: any
    player whose ``eligible_roles`` intersects the set can occupy the
    slot.
    """

    roles: frozenset[str]
    count: int
    label: str | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"SlotRequirement.count must be >= 1, got {self.count}")
        if not self.roles:
            raise ValueError("SlotRequirement.roles must be non-empty")
        unknown = self.roles - set(ALL_ROLES)
        if unknown:
            raise ValueError(
                f"SlotRequirement.roles contain unknown Mantra codes: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class MantraFormation:
    """Official Mantra Experience module (starting-XI shape)."""

    label: str
    slots: tuple[SlotRequirement, ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("MantraFormation.label must be non-empty")
        if not self.slots:
            raise ValueError("MantraFormation.slots must be non-empty")

    @property
    def required_role_codes(self) -> frozenset[str]:
        out: set[str] = set()
        for s in self.slots:
            out |= s.roles
        return frozenset(out)

    @property
    def min_outfield_players(self) -> int:
        return sum(s.count for s in self.slots)


@dataclass(frozen=True)
class FormationCoverage:
    """Result of evaluating a squad against one MantraFormation."""

    label: str
    feasible: bool
    deficits: dict[str, int]  # slot_label -> missing count (0 omitted)
    # Optional debug assignment: slot_label -> list of player_ids (or role codes)
    assigned: dict[str, list[str]] | None = None


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def _slot(
    roles: str | Iterable[str],
    count: int = 1,
    label: str | None = None,
) -> SlotRequirement:
    """Convenience: ``_slot("Dc")``, ``_slot({"E","W"}, 2, "E/W")``."""
    if isinstance(roles, str):
        role_set = frozenset({roles})
        disp = label or roles
    else:
        role_set = frozenset(roles)
        disp = label or "/".join(sorted(role_set))
    return SlotRequirement(roles=role_set, count=count, label=disp)


def _formation(label: str, *slots: SlotRequirement) -> MantraFormation:
    return MantraFormation(label=label, slots=tuple(slots))


# ---------------------------------------------------------------------------
# Official catalog (Mantra Experience 2026/27) — skeleton pending Phase-0 sign-off
# ---------------------------------------------------------------------------
# Por is treated as an implicit fixed requirement (1 from the 3-quota Por pool)
# and is *not* listed in the outfield slots.  evaluate_coverage still checks
# that at least one Por is present in the squad.
#
# Role-set notation follows the design plan + public descriptions:
#   DC/B  → {Dc, B}   (third central in a back-3)
#   E/W   → {E, W}
#   M/C   → {M, C}
#   W/A   → {W, A}
#   T/A   → {T, A}
#   A/Pc  → {A, Pc}
#   C/T   → {C, T}
#   W/T   → {W, T}
#   T/A/Pc→ {T, A, Pc}
#
# Frozen in docs/mantra_formations_2026_27.md —
# domain-owner sign-off.  Version the constant once frozen.

MANTRA_FORMATIONS: tuple[MantraFormation, ...] = (
    # 3-4-3
    # DEF: Dc×2 + {Dc,B}×1 | MID: E×2 + {M,C}×1 + C×1 | ATT: {W,A}×2 + {A,Pc}×1
    _formation(
        "3-4-3",
        _slot("Dc", 2, "Dc"),
        _slot({"Dc", "B"}, 1, "DC/B"),
        _slot("E", 2, "E"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("C", 1, "C"),
        _slot({"W", "A"}, 2, "W/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 3-4-1-2
    # DEF: Dc×2 + {Dc,B}×1 | MID: E×2 + {M,C}×1 + C×1 | TRQ: T×1 | ATT: {A,Pc}×2
    _formation(
        "3-4-1-2",
        _slot("Dc", 2, "Dc"),
        _slot({"Dc", "B"}, 1, "DC/B"),
        _slot("E", 2, "E"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("C", 1, "C"),
        _slot("T", 1, "T"),
        _slot({"A", "Pc"}, 2, "A/Pc"),
    ),
    # 3-4-2-1
    # DEF: Dc×2 + {Dc,B}×1 | MID: M×1 + {M,C}×1 + E×1 + {E,W}×1 | TRQ: T×1 + {T,A}×1 | ATT: {A,Pc}×1
    _formation(
        "3-4-2-1",
        _slot("Dc", 2, "Dc"),
        _slot({"Dc", "B"}, 1, "DC/B"),
        _slot("M", 1, "M"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("E", 1, "E"),
        _slot({"E", "W"}, 1, "E/W"),
        _slot("T", 1, "T"),
        _slot({"T", "A"}, 1, "T/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 3-5-2
    # DEF: Dc×2 + {Dc,B}×1 | MID: M×1 + {M,C}×1 + C×1 + E×1 + {E,W}×1 | ATT: {A,Pc}×2
    _formation(
        "3-5-2",
        _slot("Dc", 2, "Dc"),
        _slot({"Dc", "B"}, 1, "DC/B"),
        _slot("M", 1, "M"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("C", 1, "C"),
        _slot("E", 1, "E"),
        _slot({"E", "W"}, 1, "E/W"),
        _slot({"A", "Pc"}, 2, "A/Pc"),
    ),
    # 3-5-1-1
    # DEF: Dc×2 + {Dc,B}×1 | MID: M×2 + C×1 + {E,W}×2 | TRQ: {T,A}×1 | ATT: {A,Pc}×1
    _formation(
        "3-5-1-1",
        _slot("Dc", 2, "Dc"),
        _slot({"Dc", "B"}, 1, "DC/B"),
        _slot("M", 2, "M"),
        _slot("C", 1, "C"),
        _slot({"E", "W"}, 2, "E/W"),
        _slot({"T", "A"}, 1, "T/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 4-3-3
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: {M,C}×1 + M×1 + C×1 | ATT: {W,A}×2 + {A,Pc}×1
    _formation(
        "4-3-3",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("M", 1, "M"),
        _slot("C", 1, "C"),
        _slot({"W", "A"}, 2, "W/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 4-3-1-2
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: {M,C}×1 + M×1 + C×1 | TRQ: T×1 | ATT: {T,A,Pc}×1 + {A,Pc}×1
    _formation(
        "4-3-1-2",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("M", 1, "M"),
        _slot("C", 1, "C"),
        _slot("T", 1, "T"),
        _slot({"T", "A", "Pc"}, 1, "T/A/Pc"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 4-4-2
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: {M,C}×1 + C×1 + E×1 + {E,W}×1 | ATT: {A,Pc}×2
    _formation(
        "4-4-2",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot("C", 1, "C"),
        _slot("E", 1, "E"),
        _slot({"E", "W"}, 1, "E/W"),
        _slot({"A", "Pc"}, 2, "A/Pc"),
    ),
    # 4-1-4-1
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: M×1 + {C,T}×1 + T×1 + {E,W}×1 + W×1 | ATT: {A,Pc}×1
    _formation(
        "4-1-4-1",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot("M", 1, "M"),
        _slot({"C", "T"}, 1, "C/T"),
        _slot("T", 1, "T"),
        _slot({"E", "W"}, 1, "E/W"),
        _slot("W", 1, "W"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 4-4-1-1
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: M×1 + C×1 + {E,W}×2 | TRQ: {T,A}×1 | ATT: {A,Pc}×1
    _formation(
        "4-4-1-1",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot("M", 1, "M"),
        _slot("C", 1, "C"),
        _slot({"E", "W"}, 2, "E/W"),
        _slot({"T", "A"}, 1, "T/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
    # 4-2-3-1
    # DEF: Dd×1 + Dc×2 + Ds×1 | MID: M×1 + {M,C}×1 | TRQ: {W,T}×1 + T×1 + {W,A}×1 | ATT: {A,Pc}×1
    _formation(
        "4-2-3-1",
        _slot("Dd", 1, "Dd"),
        _slot("Dc", 2, "Dc"),
        _slot("Ds", 1, "Ds"),
        _slot("M", 1, "M"),
        _slot({"M", "C"}, 1, "M/C"),
        _slot({"W", "T"}, 1, "W/T"),
        _slot("T", 1, "T"),
        _slot({"W", "A"}, 1, "W/A"),
        _slot({"A", "Pc"}, 1, "A/Pc"),
    ),
)

MANTRA_FORMATIONS_BY_LABEL: dict[str, MantraFormation] = {
    f.label: f for f in MANTRA_FORMATIONS
}


def get_formation(label: str) -> MantraFormation:
    """Return the catalog entry or raise ``KeyError``."""
    try:
        return MANTRA_FORMATIONS_BY_LABEL[label]
    except KeyError as exc:
        known = ", ".join(sorted(MANTRA_FORMATIONS_BY_LABEL))
        raise KeyError(f"Unknown Mantra formation {label!r}. Known: {known}") from exc


# ---------------------------------------------------------------------------
# Coverage evaluation (pure)
# ---------------------------------------------------------------------------


def _player_can_fill(eligible: frozenset[str] | set[str], slot: SlotRequirement) -> bool:
    return bool(eligible & slot.roles)


def _max_matching(
    candidates: list[tuple[str, frozenset[str]]],
    slots: Sequence[SlotRequirement],
) -> tuple[dict[int, int], dict[str, int]]:
    """Exact bipartite matching: players ↔ expanded slot instances.

    Each SlotRequirement with ``count=k`` is expanded into *k* identical
    slot instances.  Returns (player_idx → slot_instance_idx, deficits).
    """
    # Expand slots into individual instances: (slot_idx, instance_in_slot)
    instances: list[tuple[int, int, str]] = []  # (slot_idx, inst, key)
    for si, slot in enumerate(slots):
        key = slot.label or "/".join(sorted(slot.roles))
        for j in range(slot.count):
            instances.append((si, j, key))

    n_inst = len(instances)
    # adjacency: player → list of instance indices it can fill
    adj: list[list[int]] = [[] for _ in candidates]
    for pi, (_, elig) in enumerate(candidates):
        for ii, (si, _, _) in enumerate(instances):
            if _player_can_fill(elig, slots[si]):
                adj[pi].append(ii)

    # match_inst[ii] = player index currently assigned to instance ii, or -1
    match_inst: list[int] = [-1] * n_inst

    def dfs(pi: int, seen: list[bool]) -> bool:
        for ii in adj[pi]:
            if seen[ii]:
                continue
            seen[ii] = True
            if match_inst[ii] == -1 or dfs(match_inst[ii], seen):
                match_inst[ii] = pi
                return True
        return False

    for pi in range(len(candidates)):
        dfs(pi, [False] * n_inst)

    # Build player → instance and per-slot deficits
    player_to_inst: dict[int, int] = {}
    filled_per_slot: dict[int, int] = {si: 0 for si in range(len(slots))}
    for ii, pi in enumerate(match_inst):
        if pi >= 0:
            player_to_inst[pi] = ii
            si = instances[ii][0]
            filled_per_slot[si] += 1

    deficits: dict[str, int] = {}
    for si, slot in enumerate(slots):
        missing = slot.count - filled_per_slot[si]
        if missing > 0:
            key = slot.label or "/".join(sorted(slot.roles))
            deficits[key] = deficits.get(key, 0) + missing

    return player_to_inst, deficits


def evaluate_coverage(
    players: Sequence[object],
    formation: MantraFormation,
    *,
    require_por: bool = True,
    return_assignment: bool = False,
) -> FormationCoverage:
    """Evaluate whether *players* can field *formation*.

    Parameters
    ----------
    players:
        Sequence of objects that expose ``eligible_roles`` (frozenset/set of
        str) and optionally ``player_id`` (str).  Classic players without
        ``eligible_roles`` (or with an empty set) are ignored for Mantra
        coverage.
    formation:
        Target MantraFormation.
    require_por:
        If True (default), the squad must contain at least one player that
        can fill Por.
    return_assignment:
        If True, populate ``FormationCoverage.assigned`` with a concrete
        mapping when feasible.

    Returns
    -------
    FormationCoverage with ``feasible`` True iff every slot can be filled
    (and Por is present when required).  Deficits map slot labels to the
    number of missing players for that group.
    """
    # Collect usable outfield candidates: (id_or_index, eligible_roles)
    candidates: list[tuple[str, frozenset[str]]] = []
    has_por = False
    for i, p in enumerate(players):
        elig = getattr(p, "eligible_roles", None)
        if elig is None:
            continue
        elig_fs = frozenset(elig) if not isinstance(elig, frozenset) else elig
        if not elig_fs:
            continue
        pid = str(getattr(p, "player_id", i))
        if "Por" in elig_fs:
            has_por = True
            # Por does not fill outfield slots
            continue
        candidates.append((pid, elig_fs))

    player_to_inst, deficits = _max_matching(candidates, formation.slots)

    por_ok = (not require_por) or has_por
    if not por_ok:
        deficits = {**deficits, "Por": 1}

    feasible = len(deficits) == 0 and por_ok

    assigned: dict[str, list[str]] | None = None
    if return_assignment and feasible:
        # Rebuild slot_key → player_ids from the matching
        assigned = {}
        # instances order matches the expansion in _max_matching
        inst_idx = 0
        for slot in formation.slots:
            key = slot.label or "/".join(sorted(slot.roles))
            filled: list[str] = []
            for _ in range(slot.count):
                # find which player (if any) was matched to this instance
                for pi, ii in player_to_inst.items():
                    if ii == inst_idx:
                        filled.append(candidates[pi][0])
                        break
                inst_idx += 1
            if filled:
                assigned[key] = filled

    return FormationCoverage(
        label=formation.label,
        feasible=feasible,
        deficits=deficits,
        assigned=assigned,
    )


def evaluate_all_coverages(
    players: Sequence[object],
    formations: Sequence[MantraFormation] | None = None,
    *,
    require_por: bool = True,
) -> dict[str, FormationCoverage]:
    """Evaluate the squad against every formation in the (or given) catalog."""
    cats = formations if formations is not None else MANTRA_FORMATIONS
    return {
        f.label: evaluate_coverage(players, f, require_por=require_por)
        for f in cats
    }
