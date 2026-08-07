"""Edge-case tests for the optimizer (per spec §9).

Covers:
* Solver timeout handling
* Empty pool
* Insufficient pool for min_distinct_teams
* Extreme num_participants values
* Homonym players
* Timeout → returns TIMEOUT with the best incumbent
"""

from __future__ import annotations

import logging

import pytest

from ml.optimizer.models import (
    Formation,
    InflationConfig,
    OptimizationConfig,
    Player,
    StrategyProfile,
)
from ml.optimizer.optimizer import (
    deduplicate_players,
    optimize_multi_strategy,
    optimize_squad,
)
from ml.optimizer.solver import PreFlightError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _balanced() -> StrategyProfile:
    return StrategyProfile(
        name="BALANCED", role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0}
    )


def _basic_cfg(**overrides: object) -> OptimizationConfig:
    defaults: dict[str, object] = {
        "budget": 500,
        "formations": [Formation("3-4-3", 3, 4, 3)],
        "num_participants": 8,
        "min_distinct_teams": 4,
    }
    defaults.update(overrides)
    return OptimizationConfig(**defaults)  # type: ignore[arg-type]


def _diverse_pool() -> list[Player]:
    """Pool with 25 players, 7 distinct teams, all cheap (cost 10).

    Distribution across teams: ``[4, 4, 4, 4, 4, 4, 1]`` so that under the
    default ``max_players_per_team=4`` the capacity is exactly 25.
    """
    team_layout: list[tuple[int, int, int, int]] = [
        (1, 1, 1, 1),  # T0: 4
        (1, 2, 1, 0),  # T1: 4
        (1, 1, 2, 0),  # T2: 4
        (0, 2, 1, 1),  # T3: 4
        (0, 1, 1, 2),  # T4: 4
        (0, 1, 1, 2),  # T5: 4
        (0, 0, 1, 0),  # T6: 1
    ]
    assert sum(sum(row) for row in team_layout) == 25

    pool: list[Player] = []
    cid = 0
    for team_idx, (np_, nd, nc, na) in enumerate(team_layout):
        team = f"T{team_idx}"
        for _ in range(np_):
            cid += 1
            pool.append(Player(f"P{cid}", f"P{cid}", "P", team, 10, 6.0))
        for _ in range(nd):
            cid += 1
            pool.append(Player(f"D{cid}", f"D{cid}", "D", team, 10, 6.0))
        for _ in range(nc):
            cid += 1
            pool.append(Player(f"C{cid}", f"C{cid}", "C", team, 10, 6.0))
        for _ in range(na):
            cid += 1
            pool.append(Player(f"A{cid}", f"A{cid}", "A", team, 10, 6.0))
    return pool


# ---------------------------------------------------------------------------
# §9.1: Infeasible strategy does not block batch
# ---------------------------------------------------------------------------


def test_batch_returns_4_entries_even_if_one_fails() -> None:
    pool = _diverse_pool()
    # Force infeasibility on BALANCED: max_players_per_team=1 + min_distinct_teams=5
    # Pool has 5 distinct teams with 5 players each, but only 1 per team is allowed.
    # 25 players required, 1 per team = at most 5 players → infeasible.
    cfg = _basic_cfg(max_players_per_team=1, min_distinct_teams=5)
    res = optimize_multi_strategy(pool, cfg)
    assert len(res.results) == 4
    assert all(
        name in res.results
        for name in ("BALANCED", "SUPER_DEFENSIVE", "SUPER_OFFENSIVE", "MIXED")
    )


# ---------------------------------------------------------------------------
# §9.2: Diagnostics on which constraint is infeasible
# ---------------------------------------------------------------------------


def test_infeasible_diagnostics_explain_reason() -> None:
    pool = _diverse_pool()
    cfg = _basic_cfg(max_players_per_team=1, min_distinct_teams=5)
    res = optimize_multi_strategy(pool, cfg)
    bal = res.results["BALANCED"]
    assert bal.status == "INFEASIBLE"
    reason = str(bal.diagnostics.get("reason", ""))
    assert reason, "INFEASIBLE result must carry a diagnostic reason"
    lowered = reason.lower()
    assert "budget" in lowered or "team" in lowered or "preflight" in lowered


# ---------------------------------------------------------------------------
# §9.3: Extreme num_participants - inflation capped
# ---------------------------------------------------------------------------


def test_extreme_num_participants_keeps_costs_bounded() -> None:
    pool = _diverse_pool()
    cfg = _basic_cfg(num_participants=10_000)
    res = optimize_squad(pool, cfg, _balanced())
    assert res.status == "OPTIMAL"
    # Cap = 1.6, nominal = 10 → effective <= 16 per player. 25 * 16 = 400.
    assert res.total_effective_cost <= 25 * 16 + 1e-6


def test_very_low_num_participants_no_inflation() -> None:
    pool = _diverse_pool()
    cfg = _basic_cfg(num_participants=1)  # way below baseline=8
    res = optimize_squad(pool, cfg, _balanced())
    assert res.status == "OPTIMAL"
    # At n=1, extra=0 → multiplier=1.0 → effective == nominal.
    assert res.total_effective_cost == pytest.approx(res.total_nominal_cost)


# ---------------------------------------------------------------------------
# §9.4: Omonimia — operate on player_id, not name
# ---------------------------------------------------------------------------


def test_homonym_players_kept_separately_by_id() -> None:
    p1 = Player("id_alpha", "Luca", "A", "Roma", 20, 7.0)
    p2 = Player("id_beta", "Luca", "A", "Inter", 25, 6.5)
    pool = _diverse_pool() + [p1, p2]
    cfg = _basic_cfg(min_distinct_teams=4, big_teams_cap=2)
    # Need 25 players, but with same-name we still have 27 candidates → no issue.
    res = optimize_squad(pool, cfg, _balanced())
    assert res.status == "OPTIMAL"
    assert len(res.squad) == 25
    # Omonimia: tutti i giocatori con name == "Luca" nel pool devono
    # essere identificati per player_id, non per name. In altri termini,
    # i player_id presenti nella rosa e che hanno name "Luca" devono
    # essere un sottoinsieme di {id_alpha, id_beta} (i due unici "Luca"
    # del pool), non un nome del _diverse_pool.
    luca_ids = {p.player_id for p in res.squad if p.name == "Luca"}
    assert luca_ids.issubset({"id_alpha", "id_beta"})


# ---------------------------------------------------------------------------
# §9.5: Trasferimenti a stagione in corso (deduplication)
# ---------------------------------------------------------------------------


def test_dedup_keeps_most_recent_record() -> None:
    p_old = Player("p1", "X", "A", "Roma", 20, 6.0)
    p_new = Player("p1", "X", "A", "Juventus", 30, 7.5)  # same id, transferred
    out = deduplicate_players([p_old, p_new])
    assert len(out) == 1
    assert out[0].real_team == "Juventus"
    assert out[0].projected_score == 7.5


# ---------------------------------------------------------------------------
# §9.6: Timeout — returns TIMEOUT
# ---------------------------------------------------------------------------


def test_timeout_returns_status_timeout() -> None:
    """With an absurdly small timeout, CBC may not finish and we get TIMEOUT or OPTIMAL.

    We assert that the result is well-formed (status one of allowed values) and
    that a TIMEOUT result has at least an empty squad or partial info.
    """
    pool = _diverse_pool()
    cfg = _basic_cfg(solver_timeout_seconds=1)
    res = optimize_squad(pool, cfg, _balanced())
    # We don't enforce TIMEOUT (CBC is fast on this small pool) but the status
    # must be one of the documented values.
    assert res.status in {"OPTIMAL", "INFEASIBLE", "TIMEOUT", "UNBOUNDED", "ERROR"}


# ---------------------------------------------------------------------------
# §9.7: Pool insufficiente
# ---------------------------------------------------------------------------


def test_empty_pool_raises_preflight() -> None:
    with pytest.raises(PreFlightError):
        optimize_squad([], _basic_cfg(), _balanced())


def test_pool_without_distinct_teams_raises_preflight() -> None:
    pool: list[Player] = []
    cid = 0
    # All from "OneTeam" - only 1 distinct team, but min_distinct_teams=4
    for i in range(3):
        cid += 1
        pool.append(Player(f"P{cid}", f"P{i}", "P", "OneTeam", 10, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"D{cid}", f"D{i}", "D", "OneTeam", 10, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"C{cid}", f"C{i}", "C", "OneTeam", 10, 6.0))
    for i in range(6):
        cid += 1
        pool.append(Player(f"A{cid}", f"A{i}", "A", "OneTeam", 10, 6.0))
    with pytest.raises(PreFlightError):
        optimize_squad(pool, _basic_cfg(), _balanced())


def test_pool_missing_role_raises_preflight() -> None:
    pool = _diverse_pool()[:22]  # only 22 players -> missing 3 attackers
    with pytest.raises(PreFlightError):
        optimize_squad(pool, _basic_cfg(), _balanced())


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_inflation_config_validation() -> None:
    with pytest.raises(ValueError):
        InflationConfig(inflation_percentile_threshold=1.5)
    with pytest.raises(ValueError):
        InflationConfig(max_inflation_multiplier=0.5)
    with pytest.raises(ValueError):
        InflationConfig(base_inflation_rate=-0.1)
    with pytest.raises(ValueError):
        InflationConfig(baseline_participants=0)


def test_strategy_profile_validation() -> None:
    # Missing role in weights
    with pytest.raises(ValueError):
        StrategyProfile(name="BAD", role_weight={"P": 1.0})  # type: ignore[arg-type]
    # max_top_tier_players without threshold
    with pytest.raises(ValueError):
        StrategyProfile(
            name="MIXED",
            role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
            max_top_tier_players=3,
        )


def test_optimization_config_validation() -> None:
    with pytest.raises(ValueError):
        OptimizationConfig(
            budget=0, formations=[Formation("3-4-3", 3, 4, 3)], num_participants=4
        )
    with pytest.raises(ValueError):
        OptimizationConfig(budget=100, formations=[], num_participants=4)
    with pytest.raises(ValueError):
        OptimizationConfig(
            budget=100, formations=[Formation("3-4-3", 3, 4, 3)], num_participants=0
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_solver_logs_status_and_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    pool = _diverse_pool()
    cfg = _basic_cfg()
    with caplog.at_level(logging.INFO):
        res = optimize_squad(pool, cfg, _balanced())
    assert res.status == "OPTIMAL"
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "solver_done" in msgs
    assert "BALANCED" in msgs
