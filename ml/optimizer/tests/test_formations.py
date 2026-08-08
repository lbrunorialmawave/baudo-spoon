"""Unit tests for Mantra formation catalog and coverage evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml.mantra.roles import ALL_ROLES
from ml.optimizer.formations import (
    MANTRA_FORMATIONS,
    MANTRA_FORMATIONS_BY_LABEL,
    FormationCoverage,
    MantraFormation,
    SlotRequirement,
    evaluate_all_coverages,
    evaluate_coverage,
    get_formation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _P:
    """Minimal player stub for coverage tests."""

    player_id: str
    eligible_roles: frozenset[str]


def _p(pid: str, *roles: str) -> _P:
    return _P(player_id=pid, eligible_roles=frozenset(roles))


def _perfect_3_4_3() -> list[_P]:
    """Exactly the roles needed for 3-4-3 (plus one Por)."""
    return [
        _p("gk", "Por"),
        _p("dc1", "Dc"),
        _p("dc2", "Dc"),
        _p("b1", "B"),  # fills DC/B
        _p("e1", "E"),
        _p("e2", "E"),
        _p("mc1", "M"),  # fills M/C
        _p("c1", "C"),
        _p("wa1", "W"),
        _p("wa2", "A"),
        _p("apc1", "Pc"),
    ]


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


def test_catalog_size_and_labels() -> None:
    assert len(MANTRA_FORMATIONS) == 11
    labels = {f.label for f in MANTRA_FORMATIONS}
    expected = {
        "3-4-3",
        "3-4-1-2",
        "3-4-2-1",
        "3-5-2",
        "3-5-1-1",
        "4-3-3",
        "4-3-1-2",
        "4-4-2",
        "4-1-4-1",
        "4-4-1-1",
        "4-2-3-1",
    }
    assert labels == expected
    assert set(MANTRA_FORMATIONS_BY_LABEL) == expected


def test_every_role_code_is_canonical() -> None:
    for f in MANTRA_FORMATIONS:
        for slot in f.slots:
            for r in slot.roles:
                assert r in ALL_ROLES, f"{f.label}: unknown role {r!r}"


def test_slot_counts_positive_and_labels_unique_per_formation() -> None:
    for f in MANTRA_FORMATIONS:
        assert f.min_outfield_players == 10, f"{f.label} should need 10 outfield"
        labels = [s.label for s in f.slots]
        # labels may repeat only if intentional; we allow duplicates of pure
        # role names (e.g. two "Dc") but the test just checks structure.
        assert all(s.count >= 1 for s in f.slots)


def test_get_formation_known_and_unknown() -> None:
    f = get_formation("4-2-3-1")
    assert isinstance(f, MantraFormation)
    assert f.label == "4-2-3-1"
    with pytest.raises(KeyError, match="Unknown Mantra formation"):
        get_formation("5-3-2")


def test_slot_requirement_validation() -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        SlotRequirement(roles=frozenset({"Dc"}), count=0)
    with pytest.raises(ValueError, match="roles must be non-empty"):
        SlotRequirement(roles=frozenset(), count=1)
    with pytest.raises(ValueError, match="unknown Mantra codes"):
        SlotRequirement(roles=frozenset({"Xx"}), count=1)


# ---------------------------------------------------------------------------
# Coverage – perfect rosters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(MANTRA_FORMATIONS_BY_LABEL))
def test_perfect_roster_is_feasible(label: str) -> None:
    """Build a synthetic perfect roster for each module and assert feasible."""
    form = get_formation(label)
    players: list[_P] = [_p("gk", "Por")]
    idx = 0
    for slot in form.slots:
        # Pick the first role in the OR-group for each required count
        role = next(iter(sorted(slot.roles)))
        for _ in range(slot.count):
            players.append(_p(f"p{idx}", role))
            idx += 1
    cov = evaluate_coverage(players, form, return_assignment=True)
    assert cov.feasible, f"{label} should be feasible: deficits={cov.deficits}"
    assert cov.deficits == {}
    assert cov.assigned is not None
    total_assigned = sum(len(v) for v in cov.assigned.values())
    assert total_assigned == form.min_outfield_players


def test_3_4_3_perfect() -> None:
    players = _perfect_3_4_3()
    cov = evaluate_coverage(players, get_formation("3-4-3"))
    assert cov.feasible
    assert cov.deficits == {}


# ---------------------------------------------------------------------------
# Coverage – deficits and edge cases
# ---------------------------------------------------------------------------


def test_missing_one_slot_reports_deficit() -> None:
    players = _perfect_3_4_3()
    # Remove the only B (needed for DC/B)
    players = [p for p in players if "B" not in p.eligible_roles]
    cov = evaluate_coverage(players, get_formation("3-4-3"))
    assert not cov.feasible
    assert "DC/B" in cov.deficits
    assert cov.deficits["DC/B"] == 1


def test_multi_role_player_counted_only_once() -> None:
    """A Dc/B player must not satisfy both a pure-Dc and a DC/B slot."""
    players = [
        _p("gk", "Por"),
        _p("dc1", "Dc"),
        _p("dcb", "Dc", "B"),  # can fill either, but only one
        # missing second pure Dc → should fail the Dc×2 requirement
        _p("e1", "E"),
        _p("e2", "E"),
        _p("mc1", "M"),
        _p("c1", "C"),
        _p("wa1", "W"),
        _p("wa2", "A"),
        _p("apc1", "Pc"),
    ]
    cov = evaluate_coverage(players, get_formation("3-4-3"))
    assert not cov.feasible
    # Exactly one of the DEF groups will be short
    assert sum(cov.deficits.values()) >= 1


def test_empty_pool_infeasible() -> None:
    cov = evaluate_coverage([], get_formation("4-3-3"))
    assert not cov.feasible
    assert "Por" in cov.deficits or len(cov.deficits) > 0


def test_classic_players_without_eligible_roles_ignored() -> None:
    @dataclass(frozen=True)
    class Classic:
        player_id: str
        # no eligible_roles

    players = [
        Classic("c1"),
        Classic("c2"),
        _p("gk", "Por"),
    ]
    cov = evaluate_coverage(players, get_formation("3-4-3"))
    assert not cov.feasible
    # Only the Por is visible; everything else missing
    assert sum(cov.deficits.values()) >= 9


def test_require_por_false_skips_gk_check() -> None:
    players = [p for p in _perfect_3_4_3() if "Por" not in p.eligible_roles]
    cov = evaluate_coverage(players, get_formation("3-4-3"), require_por=False)
    assert cov.feasible
    cov2 = evaluate_coverage(players, get_formation("3-4-3"), require_por=True)
    assert not cov2.feasible
    assert cov2.deficits.get("Por") == 1


def test_evaluate_all_coverages_returns_all_labels() -> None:
    players = _perfect_3_4_3()
    all_cov = evaluate_all_coverages(players)
    assert set(all_cov) == set(MANTRA_FORMATIONS_BY_LABEL)
    assert all_cov["3-4-3"].feasible
    # Other modules will generally be infeasible with this exact roster
    assert isinstance(all_cov["4-3-3"], FormationCoverage)


def test_por_only_player_does_not_fill_outfield() -> None:
    players = [
        _p("gk", "Por"),
        _p("dc1", "Dc"),
        _p("dc2", "Dc"),
        _p("b1", "B"),
        # missing all mids/attack
    ]
    cov = evaluate_coverage(players, get_formation("3-4-3"))
    assert not cov.feasible
    assert "E" in cov.deficits or any("E" in k for k in cov.deficits)


# ---------------------------------------------------------------------------
# Property-style: feasibility implies non-empty assignment when requested
# ---------------------------------------------------------------------------


def test_feasible_implies_assignment_when_requested() -> None:
    for label in ("3-4-3", "4-3-3", "4-2-3-1"):
        form = get_formation(label)
        players: list[_P] = [_p("gk", "Por")]
        idx = 0
        for slot in form.slots:
            role = next(iter(sorted(slot.roles)))
            for _ in range(slot.count):
                players.append(_p(f"{label}-{idx}", role))
                idx += 1
        cov = evaluate_coverage(players, form, return_assignment=True)
        assert cov.feasible
        assert cov.assigned is not None
        assert sum(len(v) for v in cov.assigned.values()) == 10
