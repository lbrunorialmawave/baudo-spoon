"""Tests for MANTRA ruleset support in the optimizer."""

from __future__ import annotations

import pytest

from ml.optimizer.models import (
    Formation,
    MANTRA_DEFAULT_QUOTAS,
    OptimizationConfig,
    Player,
    StrategyProfile,
    TOTAL_SQUAD_SIZE,
)
from ml.optimizer.optimizer import optimize_squad
from ml.optimizer.solver import PreFlightError, _preflight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _balanced_strategy() -> StrategyProfile:
    return StrategyProfile(
        name="BALANCED",
        role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
    )


def _mantra_cfg(**overrides: object) -> OptimizationConfig:
    defaults: dict[str, object] = {
        "budget": 500,
        "formations": [Formation("4-3-3", 4, 3, 3)],
        "num_participants": 8,
        "min_distinct_teams": 4,
        "max_players_per_team": 25,  # large cap so team constraint doesn't interfere with small test pools
        "ruleset": "MANTRA",
    }
    defaults.update(overrides)
    return OptimizationConfig(**defaults)  # type: ignore[arg-type]


def _make_player(
    pid: str,
    classic_role: str,
    team: str,
    eligible: frozenset[str],
    score: float = 6.0,
    cost: int = 10,
) -> Player:
    return Player(
        player_id=pid,
        name=pid,
        role=classic_role,  # type: ignore[arg-type]
        real_team=team,
        cost=cost,
        projected_score=score,
        eligible_roles=eligible,
    )


def _mantra_pool() -> list[Player]:
    """Pool of 25 players covering MANTRA_DEFAULT_QUOTAS across 5 teams.

    Quotas: Por=3, Dc=3, B=2, Dd=2, Ds=1, E=1, M=2, C=5, T=1, W=1, A=2, Pc=2 = 25
    Each player has a single eligible_role matching the quota slot.
    """
    quota_map = [
        ("Por", "P", "Por"),
        ("Por", "P", "Por"),
        ("Por", "P", "Por"),
        ("Dc", "D", "Dc"),
        ("Dc", "D", "Dc"),
        ("Dc", "D", "Dc"),
        ("B", "D", "B"),
        ("B", "D", "B"),
        ("Dd", "D", "Dd"),
        ("Dd", "D", "Dd"),
        ("Ds", "D", "Ds"),
        ("E", "C", "E"),
        ("M", "C", "M"),
        ("M", "C", "M"),
        ("C", "C", "C"),
        ("C", "C", "C"),
        ("C", "C", "C"),
        ("C", "C", "C"),
        ("C", "C", "C"),
        ("T", "A", "T"),
        ("W", "A", "W"),
        ("A", "A", "A"),
        ("A", "A", "A"),
        ("Pc", "A", "Pc"),
        ("Pc", "A", "Pc"),
    ]
    assert len(quota_map) == TOTAL_SQUAD_SIZE
    teams = ["T0", "T1", "T2", "T3", "T4"]
    pool = []
    for i, (prefix, classic, mantra) in enumerate(quota_map):
        team = teams[i % len(teams)]
        pool.append(
            _make_player(
                pid=f"{prefix}_{i}",
                classic_role=classic,
                team=team,
                eligible=frozenset([mantra]),
            )
        )
    return pool


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------


def test_mantra_preflight_fails_without_eligible_roles() -> None:
    """Players missing eligible_roles trigger preflight failure in MANTRA mode."""
    pool = _mantra_pool()
    # Strip eligible_roles from the first player
    broken = Player(
        player_id=pool[0].player_id,
        name=pool[0].name,
        role=pool[0].role,
        real_team=pool[0].real_team,
        cost=pool[0].cost,
        projected_score=pool[0].projected_score,
        eligible_roles=frozenset(),
    )
    modified_pool = [broken] + pool[1:]
    config = _mantra_cfg()
    with pytest.raises(PreFlightError, match="eligible_roles"):
        _preflight(modified_pool, config)


def test_mantra_preflight_fails_insufficient_coverage() -> None:
    """Preflight fails when a role quota cannot be covered by the pool."""
    pool = _mantra_pool()
    # Remove all Por players (first 3)
    no_por = [p for p in pool if "Por" not in p.eligible_roles]
    config = _mantra_cfg()
    with pytest.raises(PreFlightError, match="Por"):
        _preflight(no_por, config)


def test_classic_preflight_unchanged() -> None:
    """Classic preflight still works after MANTRA code was added."""
    from ml.optimizer.tests.test_edge_cases import _diverse_pool, _basic_cfg
    pool = _diverse_pool()
    config = _basic_cfg()
    # Should not raise
    _preflight(pool, config)


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------


def test_mantra_solve_respects_role_quotas() -> None:
    """MANTRA solve produces a squad satisfying all 12-role quotas."""
    pool = _mantra_pool()
    # Add extra players so the solver has some freedom
    extras = [
        _make_player(f"extra_{i}", "C", f"T{i % 5}", frozenset(["C"]))
        for i in range(10)
    ]
    pool = pool + extras
    config = _mantra_cfg()
    result = optimize_squad(pool, config, _balanced_strategy())

    assert result.status == "OPTIMAL"
    assert len(result.squad) == TOTAL_SQUAD_SIZE

    # Count each mantra role in selected squad
    counts: dict[str, int] = {}
    for p in result.squad:
        selected_player = next(sp for sp in pool if sp.player_id == p.player_id)
        for r in selected_player.eligible_roles:
            counts[r] = counts.get(r, 0) + 1

    expected = config.mantra_role_quotas or MANTRA_DEFAULT_QUOTAS
    for role, quota in expected.items():
        if quota > 0:
            assert counts.get(role, 0) == quota, (
                f"Role {role}: expected {quota}, got {counts.get(role, 0)}"
            )


def test_mantra_polivalent_player_fills_one_slot() -> None:
    """A player with two eligible roles is selected into at most one slot."""
    pool = _mantra_pool()
    # Make the first Dc player also eligible for B
    original = pool[3]  # Dc_3
    polivalent = Player(
        player_id=original.player_id,
        name=original.name,
        role=original.role,
        real_team=original.real_team,
        cost=original.cost,
        projected_score=original.projected_score,
        eligible_roles=frozenset(["Dc", "B"]),
    )
    pool[3] = polivalent
    config = _mantra_cfg()
    result = optimize_squad(pool, config, _balanced_strategy())

    assert result.status == "OPTIMAL"
    # The polivalent player appears at most once in the squad
    occurrences = sum(1 for p in result.squad if p.player_id == polivalent.player_id)
    assert occurrences <= 1


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def test_mantra_quotas_must_sum_to_squad_size() -> None:
    bad_quotas = dict(MANTRA_DEFAULT_QUOTAS)
    bad_quotas["Por"] = 99  # makes total >> 25
    with pytest.raises(ValueError, match="sum"):
        _mantra_cfg(mantra_role_quotas=bad_quotas)


def test_mantra_default_quotas_sum_to_squad_size() -> None:
    assert sum(MANTRA_DEFAULT_QUOTAS.values()) == TOTAL_SQUAD_SIZE


# ---------------------------------------------------------------------------
# Mantra formation catalog post-hoc coverage (Phase 2)
# ---------------------------------------------------------------------------


def test_mantra_result_includes_formation_coverage() -> None:
    """When ruleset=MANTRA, OptimizationResult carries mantra_formation_feasibility."""
    pool = _mantra_pool()
    config = _mantra_cfg()
    result = optimize_squad(pool, config, _balanced_strategy())

    assert result.status == "OPTIMAL"
    assert result.mantra_formation_feasibility is not None
    assert len(result.mantra_formation_feasibility) == 11
    for label, cov in result.mantra_formation_feasibility.items():
        assert cov.label == label
        assert isinstance(cov.feasible, bool)
        assert isinstance(cov.deficits, dict)


def test_classic_result_has_no_mantra_coverage() -> None:
    """CLASSIC path leaves mantra_formation_feasibility as None."""
    from ml.optimizer.models import ROLE_QUOTAS

    pool = [
        Player(
            player_id=f"p{i}",
            name=f"p{i}",
            role=role,  # type: ignore[arg-type]
            real_team=f"T{i % 5}",
            cost=10,
            projected_score=6.0,
        )
        for i, role in enumerate(
            ["P"] * 3 + ["D"] * 8 + ["C"] * 8 + ["A"] * 6
        )
    ]
    config = OptimizationConfig(
        budget=500,
        formations=[Formation("4-3-3", 4, 3, 3)],
        num_participants=8,
        min_distinct_teams=3,
        max_players_per_team=25,
        ruleset="CLASSIC",
    )
    result = optimize_squad(pool, config, _balanced_strategy())
    assert result.status == "OPTIMAL"
    assert result.mantra_formation_feasibility is None
    # Classic field still populated
    assert "4-3-3" in result.formation_feasibility


# ---------------------------------------------------------------------------
# Preferred Mantra formation hard constraint (Phase 3)
# ---------------------------------------------------------------------------


def test_enforce_preferred_mantra_makes_coverage_true() -> None:
    """Enforce 3-4-3 → selected squad must cover that module."""
    pool = _mantra_pool()
    # Enrich with multi-role players so formation slots can be filled without
    # fighting the exact role quotas (e.g. a second E via a C-quota player).
    extra = [
        _make_player("e_extra", "C", "T0", frozenset({"E", "C"}), score=7.0, cost=8),
        _make_player("w_extra", "A", "T1", frozenset({"W", "A"}), score=7.0, cost=8),
        _make_player("wa2", "A", "T2", frozenset({"W", "A"}), score=7.0, cost=8),
        _make_player("b_extra", "D", "T2", frozenset({"B", "Dc"}), score=7.0, cost=8),
        _make_player("mc_extra", "C", "T3", frozenset({"M", "C"}), score=7.0, cost=8),
    ]
    pool = pool + extra
    config = _mantra_cfg(
        preferred_mantra_formation="3-4-3",
        enforce_preferred_mantra_formation=True,
        max_players_per_team=25,
        min_distinct_teams=3,
    )
    result = optimize_squad(pool, config, _balanced_strategy())
    assert result.status == "OPTIMAL", result.diagnostics
    assert result.mantra_formation_feasibility is not None
    assert result.mantra_formation_feasibility["3-4-3"].feasible is True


def test_enforce_infeasible_module_returns_infeasible() -> None:
    """Enforce a module the pool cannot cover → PreFlightError with clear reason."""
    from ml.optimizer.solver import PreFlightError

    # Pool that satisfies relaxed quotas but has zero W/A → 3-4-3 impossible.
    pool = _mantra_pool()
    stripped = []
    for p in pool:
        elig = frozenset(r for r in p.eligible_roles if r not in ("W", "A", "Pc"))
        if not elig:
            elig = frozenset({"C"})
        stripped.append(
            Player(
                player_id=p.player_id,
                name=p.name,
                role=p.role,
                real_team=p.real_team,
                cost=p.cost,
                projected_score=p.projected_score,
                eligible_roles=elig,
            )
        )
    # Pad C coverage so quota preflight passes
    for i in range(5):
        stripped.append(
            _make_player(f"c_pad_{i}", "C", f"T{i % 4}", frozenset({"C"}), score=5.0, cost=5)
        )
    pool = stripped
    quotas = {
        "Por": 3, "Dc": 3, "B": 2, "Dd": 2, "Ds": 1,
        "E": 1, "M": 2, "C": 11,
        "T": 0, "W": 0, "A": 0, "Pc": 0,
    }
    config = _mantra_cfg(
        preferred_mantra_formation="3-4-3",
        enforce_preferred_mantra_formation=True,
        mantra_role_quotas=quotas,
        max_players_per_team=25,
        min_distinct_teams=3,
    )
    with pytest.raises(PreFlightError, match="preferred Mantra formation|3-4-3|deficits"):
        optimize_squad(pool, config, _balanced_strategy())


def test_enforce_false_does_not_hard_constrain() -> None:
    """preferred set but enforce=False → soft only; still OPTIMAL even if coverage false."""
    pool = _mantra_pool()
    config = _mantra_cfg(
        preferred_mantra_formation="3-4-3",
        enforce_preferred_mantra_formation=False,
        max_players_per_team=25,
        min_distinct_teams=3,
    )
    result = optimize_squad(pool, config, _balanced_strategy())
    assert result.status == "OPTIMAL"
    # Coverage may or may not be True depending on pool; field must exist
    assert result.mantra_formation_feasibility is not None
    assert "3-4-3" in result.mantra_formation_feasibility


def test_unknown_preferred_mantra_label_raises() -> None:
    with pytest.raises(ValueError, match="not in catalog"):
        _mantra_cfg(preferred_mantra_formation="5-3-2")


def test_enforce_without_preferred_raises() -> None:
    with pytest.raises(ValueError, match="requires preferred_mantra_formation"):
        _mantra_cfg(enforce_preferred_mantra_formation=True)
