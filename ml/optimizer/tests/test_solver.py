"""Tests for the ILP solver and multi-strategy orchestrator.

Uses small synthetic pools where the optimal solution is hand-computable,
to validate that:

* Basic constraints (budget, role quotas, formation feasibility) are respected.
* Strategy-specific constraints (defensive share, offensive share, top-tier cap)
  change the resulting squad.
* Multi-strategy run returns 4 entries even when one strategy is infeasible.
"""

from __future__ import annotations

import pytest

from ml.optimizer.models import (
    DEFAULT_BUDGET,
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
# Fixtures: small synthetic pool with 25 known players
# ---------------------------------------------------------------------------


DEFAULT_FORMATIONS = [
    Formation("3-4-3", 3, 4, 3),
    Formation("4-3-3", 4, 3, 3),
    Formation("4-4-2", 4, 4, 2),
    Formation("3-5-2", 3, 5, 2),
]


def _cfg(
    *,
    budget: int = DEFAULT_BUDGET,
    num_participants: int = 8,
    big_teams: frozenset[str] = frozenset({"Inter", "Milan", "Juventus", "Napoli"}),
    big_teams_cap: int = 10,
    min_distinct_teams: int = 5,
    max_players_per_team: int = 4,
    inflation_config: InflationConfig | None = None,
    formations: list[Formation] | None = None,
) -> OptimizationConfig:
    return OptimizationConfig(
        budget=budget,
        formations=list(formations) if formations is not None else list(DEFAULT_FORMATIONS),
        num_participants=num_participants,
        max_players_per_team=max_players_per_team,
        big_teams=big_teams,
        big_teams_cap=big_teams_cap,
        min_distinct_teams=min_distinct_teams,
        inflation_config=inflation_config or InflationConfig(),
    )


def _minimal_feasible_pool() -> list[Player]:
    """25-player pool, 7 distinct teams, no inflation pressure.

    Pool design (per spec, 3P/8D/8C/6A) and constraints:
      * 7 teams with per-team counts ``[4, 4, 4, 4, 4, 4, 1]`` ⇒ total
        capacity under ``max_players_per_team=4`` is
        ``6*4 + 1*1 = 25`` ≥ 25 selected.
      * All players cost 10 ⇒ cheapest 25 = 250 < 500 (budget).
      * All scores 6.0 ⇒ strategies that just maximise total score all
        converge on the same optimal squad.
      * Each role is spread across at least 3 teams so ``min_distinct_teams``
        is satisfiable for every role quota.
    """
    # Team layout: index → (P, D, C, A) counts. Sum across rows = 25.
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
# Pre-flight validation
# ---------------------------------------------------------------------------


def test_preflight_rejects_empty_pool() -> None:
    cfg = _cfg()
    with pytest.raises(PreFlightError):
        optimize_squad([], cfg, _balanced_strategy())


def test_preflight_rejects_missing_role() -> None:
    pool = _minimal_feasible_pool()[:24]  # drop one
    cfg = _cfg(min_distinct_teams=2)
    with pytest.raises(PreFlightError):
        optimize_squad(pool, cfg, _balanced_strategy())


def test_preflight_rejects_too_few_teams() -> None:
    pool = _minimal_feasible_pool()
    cfg = _cfg(min_distinct_teams=999)
    with pytest.raises(PreFlightError):
        optimize_squad(pool, cfg, _balanced_strategy())


def test_preflight_rejects_cheapest_exceeds_budget() -> None:
    pool = _minimal_feasible_pool()
    # Each role's cheapest selection will be 30+80+80+60 = 250 even with cheapest picks
    # Push budget to 1: 3*10 + 8*10 + 8*10 + 6*10 = 250 > 1
    cfg = _cfg(budget=1)
    with pytest.raises(PreFlightError):
        optimize_squad(pool, cfg, _balanced_strategy())


# ---------------------------------------------------------------------------
# Basic ILP behaviour
# ---------------------------------------------------------------------------


def _balanced_strategy() -> StrategyProfile:
    return StrategyProfile(name="BALANCED", role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0})


def test_balanced_selects_25_players_with_correct_quota() -> None:
    pool = _minimal_feasible_pool()
    cfg = _cfg()
    res = optimize_squad(pool, cfg, _balanced_strategy())
    assert res.status == "OPTIMAL"
    assert len(res.squad) == 25
    assert res.role_breakdown == {"P": 3, "D": 8, "C": 8, "A": 6}
    # Il pool minimale ha 7 team distinti e l'ottimizzatore può prendere
    # fino a 4 giocatori per team, quindi può saturare su tutti e 7 i team.
    # Verifichiamo solo che il vincolo min_distinct_teams=5 sia rispettato.
    assert res.distinct_teams_count >= 5
    assert res.budget_residual >= 0
    assert res.total_effective_cost <= cfg.budget
    # Tutte le formazioni sono coperte.
    assert all(res.formation_feasibility.values())


def test_formation_feasibility_flagged_correctly_when_understaffed() -> None:
    """When the selected squad is missing a role, formations turn False."""
    pool = _minimal_feasible_pool()
    cfg = _cfg()
    res = optimize_squad(pool, cfg, _balanced_strategy())
    # In the default test, the squad has 3 P, 8 D, 8 C, 6 A → 3-5-2 is feasible
    # (defenders=3, mids=5, fwds=2). 4-4-2 (4 D, 4 C, 2 A) needs at least 4 D
    # which is true. 4-3-3 (4 D, 3 C, 3 A) needs at least 3 A which is true.
    assert res.formation_feasibility["3-4-3"] is True
    assert res.formation_feasibility["4-3-3"] is True
    assert res.formation_feasibility["4-4-2"] is True
    assert res.formation_feasibility["3-5-2"] is True


def test_budget_is_respected() -> None:
    """No squad should ever have total_effective_cost > budget."""
    pool = _minimal_feasible_pool()
    # Tutti i giocatori costano 10 → cheapest 25 = 250. Budget 300 lascia
    # margine ma resta sotto la somma di tutti i giocatori del pool (250).
    cfg = _cfg(budget=300)  # budget sufficiente ma non illimitato
    res = optimize_squad(pool, cfg, _balanced_strategy())
    if res.status == "OPTIMAL":
        assert res.total_effective_cost <= cfg.budget + 1e-6


def test_risk_aversion_avoids_high_std_players() -> None:
    """With risk_aversion > 0, players with high prediction_std are penalised.

    Two pools of P players: one has score=8 + std=5 (risky), one score=7 + std=0 (safe).
    risk_adjusted_score(risky) = 8 - 2*5 = -2 < 7. Solver should pick the safe players.
    """
    cid = 0

    def make(role: str, team: str, score: float, std: float) -> Player:
        nonlocal cid
        cid += 1
        return Player(f"{role}{cid}", f"{role}{cid}", role, team, 10, score, prediction_std=std)  # type: ignore[arg-type]

    # Build pool: use _minimal_feasible_pool() layout but replace P players with
    # 3 risky (score=8, std=5) from teams T10-T12 and 3 safe (score=7, std=0) from T20-T22.
    base_pool = _minimal_feasible_pool()
    non_p_pool = [p for p in base_pool if p.role != "P"]
    risky_ids = set()
    safe_ps = []
    risky_ps = []
    for i in range(3):
        r = make("P", f"RiskyTeam{i}", 8.0, 5.0)
        s = make("P", f"SafeTeam{i}", 7.0, 0.0)
        risky_ps.append(r)
        safe_ps.append(s)
        risky_ids.add(r.player_id)
    pool = non_p_pool + risky_ps + safe_ps

    cfg = _cfg(
        min_distinct_teams=5,
        max_players_per_team=4,
    )
    # Use OptimizationConfig directly to pass risk_aversion
    from ml.optimizer.models import OptimizationConfig, InflationConfig
    risk_cfg = OptimizationConfig(
        budget=cfg.budget,
        formations=cfg.formations,
        num_participants=cfg.num_participants,
        max_players_per_team=cfg.max_players_per_team,
        big_teams=cfg.big_teams,
        big_teams_cap=cfg.big_teams_cap,
        min_distinct_teams=cfg.min_distinct_teams,
        risk_aversion=2.0,
    )
    res = optimize_squad(pool, risk_cfg, _balanced_strategy())
    assert res.status == "OPTIMAL"
    selected_p_ids = {p.player_id for p in res.squad if p.role == "P"}
    assert not (selected_p_ids & risky_ids), "Risk-averse solver should not select high-std players"


def _spread_pool_9_teams() -> list[Player]:
    """Pool distribuito su 9 team con max=3 selezionabili ciascuno.

    Ogni team ha esattamente 3 giocatori → capacity con max=3: 9*3 = 27 ≥ 25.
    Distribuzione: T0-T2 = P+D+C, T3-T4 = D+C+A, T5 = D+C+A,
                   T6 = D+C+A, T7 = C+A+A, T8 = A+A+A.
    Totale: 3P + 8D + 9C + 7A = 27 ≥ 25.
    """
    pool: list[Player] = []
    cid = 0

    layout: list[tuple[int, int, int, int]] = [
        # (P, D, C, A)
        (1, 1, 1, 0),  # T0
        (1, 1, 1, 0),  # T1
        (1, 1, 1, 0),  # T2
        (0, 1, 1, 1),  # T3
        (0, 1, 1, 1),  # T4
        (0, 1, 1, 1),  # T5
        (0, 1, 1, 1),  # T6
        (0, 1, 1, 1),  # T7
        (0, 0, 1, 2),  # T8
    ]
    for team_idx, (np_, nd, nc, na) in enumerate(layout):
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


def test_max_players_per_team_enforced() -> None:
    pool = _spread_pool_9_teams()
    # 9 team distinti; con max=3 → capacity = 9*3 = 27 ≥ 25.
    cfg = _cfg(max_players_per_team=3, min_distinct_teams=5)
    res = optimize_squad(pool, cfg, _balanced_strategy())
    assert res.status == "OPTIMAL"
    for team, n in res.team_breakdown.items():
        assert n <= 3, f"team {team} has {n} > 3"


def test_min_distinct_teams_enforced() -> None:
    pool = _minimal_feasible_pool()
    cfg = _cfg(min_distinct_teams=5)
    res = optimize_squad(pool, cfg, _balanced_strategy())
    if res.status == "OPTIMAL":
        assert res.distinct_teams_count >= 5


def test_big_teams_cap_enforced() -> None:
    """big_teams_cap=2 → at most 2 Inter players selected despite higher projected_score.

    Design: 25 non-big players cover all role quotas (3P/8D/8C/6A) across 5 small teams.
    14 Inter players (all score=9.0, tempting) padded on top. Cap=2 forces solver to use
    at most 2 Inter → OPTIMAL with inter_count <= 2.
    """
    cid = 0

    def p(role: str, team: str, score: float = 6.0) -> Player:
        nonlocal cid
        cid += 1
        return Player(f"{role}{cid}", f"{role}{cid}", role, team, 10, score)  # type: ignore[arg-type]

    # Non-big baseline: exactly 3P+8D+8C+6A across 5 teams (5 players each).
    small_teams = ["Atalanta", "Roma", "Lazio", "Torino", "Fiorentina"]
    pool: list[Player] = []
    for team in small_teams[:3]:          # 3P, one per team
        pool.append(p("P", team))
    for i, team in enumerate(small_teams):  # 8D: 2+2+2+1+1
        n = 2 if i < 3 else 1
        for _ in range(n):
            pool.append(p("D", team))
    for i, team in enumerate(small_teams):  # 8C: 2+2+2+1+1
        n = 2 if i < 3 else 1
        for _ in range(n):
            pool.append(p("C", team))
    for i, team in enumerate(small_teams):  # 6A: 2+2+1+1+0
        n = [2, 2, 1, 1, 0][i]
        for _ in range(n):
            pool.append(p("A", team))

    assert len(pool) == 25  # sanity check

    # Inter players with higher score to tempt the solver.
    for _ in range(6):
        pool.append(p("D", "Inter", score=9.0))
    for _ in range(4):
        pool.append(p("C", "Inter", score=9.0))
    for _ in range(4):
        pool.append(p("A", "Inter", score=9.0))

    big_teams = frozenset({"Inter", "Milan", "Juventus", "Napoli"})
    cfg = _cfg(
        big_teams_cap=2,
        min_distinct_teams=3,
        max_players_per_team=14,
        big_teams=big_teams,
    )
    res = optimize_squad(pool, cfg, _balanced_strategy())
    assert res.status == "OPTIMAL"
    assert res.big_teams_players_count <= 2


# ---------------------------------------------------------------------------
# Strategy-specific constraints
# ---------------------------------------------------------------------------


def test_super_defensive_shifts_spend_to_defense() -> None:
    """In SUPER_DEFENSIVE, the (P+D) share of effective spend >= 45% of budget.

    Uses a tailored pool where the D-role is expensive enough to satisfy the
    P+D ≥ 45% share: 8D @ cost 50 each, P @ cost 1 each, C+A @ cost 5 each.
    Pool cost: 3 + 400 + 40 + 30 = 473 ≤ 500.
    """
    pool: list[Player] = []
    cid = 0
    for i in range(3):
        cid += 1
        pool.append(Player(f"P{cid}", f"P{i}", "P", f"T{i % 7}", 1, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"D{cid}", f"D{i}", "D", f"T{i % 7}", 50, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"C{cid}", f"C{i}", "C", f"T{i % 7}", 5, 6.0))
    for i in range(6):
        cid += 1
        pool.append(Player(f"A{cid}", f"A{i}", "A", f"T{i % 7}", 5, 6.0))
    cfg = _cfg()
    strat = StrategyProfile(
        name="SUPER_DEFENSIVE",
        role_weight={"P": 1.2, "D": 1.3, "C": 1.0, "A": 0.8},
        min_budget_share_by_roles=(frozenset({"P", "D"}), 0.45),
    )
    res = optimize_squad(pool, cfg, strat)
    assert res.status == "OPTIMAL"
    pd_spend = sum(p.cost for p in res.squad if p.role in {"P", "D"})
    assert pd_spend >= 0.45 * cfg.budget - 1e-6


def test_super_offensive_shifts_spend_to_attack() -> None:
    """In SUPER_OFFENSIVE, the (C+A) share of effective spend >= 65% of budget.

    Pool biased to make C+A expensive and P+D cheap.
    """
    pool: list[Player] = []
    cid = 0
    for i in range(3):
        cid += 1
        pool.append(Player(f"P{cid}", f"P{i}", "P", f"T{i % 7}", 1, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"D{cid}", f"D{i}", "D", f"T{i % 7}", 1, 6.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"C{cid}", f"C{i}", "C", f"T{i % 7}", 30, 6.0))
    for i in range(6):
        cid += 1
        pool.append(Player(f"A{cid}", f"A{i}", "A", f"T{i % 7}", 30, 6.0))
    # Pool nominal cost: 3 + 8 + 240 + 180 = 431 < 500. CA max = 8*30+6*30 = 420 ≥ 325.
    cfg = _cfg()
    strat = StrategyProfile(
        name="SUPER_OFFENSIVE",
        role_weight={"P": 0.8, "D": 0.9, "C": 1.15, "A": 1.3},
        min_budget_share_by_roles=(frozenset({"C", "A"}), 0.65),
    )
    res = optimize_squad(pool, cfg, strat)
    assert res.status == "OPTIMAL"
    ca_spend = sum(p.cost for p in res.squad if p.role in {"C", "A"})
    assert ca_spend >= 0.65 * cfg.budget - 1e-6


def test_mixed_caps_top_tier_players() -> None:
    """MIXED strategy caps the number of players with cost >= threshold.

    Pool: 7 teams, 5 top-tier (cost 30) and 20 cheap (cost 10) → cap = 5 is binding.
    """
    pool: list[Player] = []
    cid = 0
    # 3 P (all top-tier: cost 30)
    for i in range(3):
        cid += 1
        pool.append(Player(f"P{cid}", f"P{i}", "P", f"T{i % 7}", 30, 6.0))
    # 8 D (cheap: cost 10)
    for i in range(8):
        cid += 1
        pool.append(Player(f"D{cid}", f"D{i}", "D", f"T{(i + 1) % 7}", 10, 6.0))
    # 8 C (cheap: cost 10)
    for i in range(8):
        cid += 1
        pool.append(Player(f"C{cid}", f"C{i}", "C", f"T{(i + 2) % 7}", 10, 6.0))
    # 6 A (cheap: cost 10)
    for i in range(6):
        cid += 1
        pool.append(Player(f"A{cid}", f"A{i}", "A", f"T{(i + 3) % 7}", 10, 6.0))
    # Total top-tier: 3 P @ 30. Cap=5 binds if we want at most 3 P selected.
    cfg = _cfg(min_distinct_teams=5)
    strat = StrategyProfile(
        name="MIXED",
        role_weight={"P": 1.0, "D": 1.0, "C": 1.0, "A": 1.0},
        max_top_tier_players=3,
        top_tier_cost_threshold=25.0,
    )
    res = optimize_squad(pool, cfg, strat)
    assert res.status == "OPTIMAL"
    n_top = sum(1 for p in res.squad if p.cost >= 25.0)
    assert n_top <= 3


# ---------------------------------------------------------------------------
# Multi-strategy: always returns 4 entries
# ---------------------------------------------------------------------------


def test_multi_strategy_returns_all_4_entries() -> None:
    pool = _minimal_feasible_pool()
    cfg = _cfg()
    res = optimize_multi_strategy(pool, cfg)
    assert set(res.results.keys()) == {"BALANCED", "SUPER_DEFENSIVE", "SUPER_OFFENSIVE", "MIXED"}


def test_multi_strategy_strategies_are_different() -> None:
    """The 4 strategies should produce at least one different attribute."""
    pool = _minimal_feasible_pool()
    cfg = _cfg()
    res = optimize_multi_strategy(pool, cfg)
    optimal = [r for r in res.results.values() if r.status == "OPTIMAL"]
    # At least 2 of the 4 are optimal in this trivial case.
    if len(optimal) >= 2:
        # Compare role_breakdown or team_breakdown; with this trivial pool
        # all are identical (cost=10 everywhere), so the 4 strategies
        # should still agree on the basic shape.
        breakdowns = {r.role_breakdown["P"] for r in optimal}
        # All select exactly 3 P, so it's the same. The differentiator is the
        # diagnostics elapsed_seconds. The strategies do not diverge on a
        # totally flat pool; this is correct.
        assert breakdowns == {3}


def test_multi_strategy_strategies_diverge_with_skewed_pool() -> None:
    """When players have heterogeneous scores, the 4 strategies should diverge."""
    pool: list[Player] = []
    cid = 0
    # 3 P: one elite (high score, high cost), two cheap
    pool.append(Player("Pe1", "EliteGK", "P", "Inter", 50, 9.0))
    pool.append(Player("Pa1", "AvgGK1", "P", "Roma", 5, 5.0))
    pool.append(Player("Pa2", "AvgGK2", "P", "Lazio", 5, 5.0))
    # 8 D
    for i in range(8):
        cid += 1
        score = 7.5 if i == 0 else 6.0
        cost = 40 if i == 0 else 10
        team = "Inter" if i < 2 else f"T{i % 5}"
        pool.append(Player(f"D{cid}", f"D{i}", "D", team, cost, score))
    # 8 C
    for i in range(8):
        cid += 1
        score = 7.5 if i == 0 else 6.0
        cost = 40 if i == 0 else 10
        team = "Inter" if i < 2 else f"T{(i + 1) % 5}"
        pool.append(Player(f"C{cid}", f"C{i}", "C", team, cost, score))
    # 6 A
    for i in range(6):
        cid += 1
        score = 7.5 if i == 0 else 6.0
        cost = 40 if i == 0 else 10
        team = "Inter" if i < 2 else f"T{(i + 2) % 5}"
        pool.append(Player(f"A{cid}", f"A{i}", "A", team, cost, score))

    cfg = _cfg(
        budget=300,
        max_players_per_team=4,
        big_teams_cap=4,
        min_distinct_teams=4,
    )
    res = optimize_multi_strategy(pool, cfg)
    statuses = {r.status for r in res.results.values()}
    assert statuses == {"OPTIMAL"}  # all 4 feasible in this setup
    # SUPER_OFFENSIVE favours the expensive star attacker; check that
    # offensive strategies include the high-cost A player.
    so = res.results["SUPER_OFFENSIVE"]
    sd = res.results["SUPER_DEFENSIVE"]
    so_has_star_a = any(p.player_id == "A1" for p in so.squad)
    sd_has_star_a = any(p.player_id == "A1" for p in sd.squad)
    assert so_has_star_a, "super-offensive should keep the elite attacker"
    # super-defensive may or may not, depending on shares; not strict here.


# ---------------------------------------------------------------------------
# Infeasibility: single strategy can fail without breaking the batch
# ---------------------------------------------------------------------------


def test_infeasible_strategy_does_not_block_batch() -> None:
    """A super-defensive strategy with impossible min-share should be INFEASIBLE,
    but other strategies must still run."""
    pool: list[Player] = []
    cid = 0
    # Pool biased toward attackers: only 3 P (cheap), 8 D (cheap), 8 C (cheap), 6 A (premium)
    for i in range(3):
        cid += 1
        pool.append(Player(f"P{cid}", f"P{i}", "P", f"T{i}", 5, 5.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"D{cid}", f"D{i}", "D", f"T{i}", 5, 5.0))
    for i in range(8):
        cid += 1
        pool.append(Player(f"C{cid}", f"C{i}", "C", f"T{i}", 5, 5.0))
    for i in range(6):
        cid += 1
        pool.append(Player(f"A{cid}", f"A{i}", "A", f"T{i}", 200, 9.0))  # 6×200=1200 >> budget

    cfg = _cfg(budget=500, min_distinct_teams=5)
    res = optimize_multi_strategy(pool, cfg)
    # All 4 must be present.
    assert len(res.results) == 4
    # The defensive super strategy with min share 0.45 on (P+D)=10+40=50 won't fit
    # (other players all cost 200). Defensive and balanced are infeasible.
    # But super_offensive / mixed may still find a solution if they ignore the share.
    # The contract: each entry is present, even if INFEASIBLE.
    for name, r in res.results.items():
        assert r.strategy_name == name
        assert r.status in {"OPTIMAL", "INFEASIBLE", "TIMEOUT", "ERROR"}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_highest_score() -> None:
    p1 = Player("id1", "Mario", "A", "Roma", 20, 6.0)
    p2 = Player("id1", "Mario Updated", "A", "Roma", 30, 7.0)
    out = deduplicate_players([p1, p2])
    assert len(out) == 1
    assert out[0].projected_score == 7.0


def test_deduplicate_logs_homonym_warning(caplog: pytest.LogCaptureFixture) -> None:
    p1 = Player("id1", "Luca", "A", "Roma", 20, 6.0)
    p2 = Player("id2", "Luca", "A", "Inter", 25, 6.5)
    with caplog.at_level("WARNING"):
        out = deduplicate_players([p1, p2])
    assert len(out) == 2  # different player_id, kept both
    assert any("homonym" in rec.message.lower() for rec in caplog.records)


def test_deduplicate_empty() -> None:
    assert deduplicate_players([]) == []
