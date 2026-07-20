"""PuLP/CBC-based ILP solver for a single strategy.

Single responsibility: build and solve a 0/1 selection problem given a
``StrategyProfile``, a pool of ``Player`` and a fully-populated
``OptimizationConfig``.  The function is fully parametric — no value
of the optimization is hardcoded here; everything comes from the configs.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import pulp

from ml.optimizer.inflation import compute_role_percentile_map, estimate_effective_cost
from ml.optimizer.models import (
    MANTRA_DEFAULT_QUOTAS,
    OptimizationConfig,
    OptimizationResult,
    Player,
    ROLE_QUOTAS,
    SOLVER_STATUS_ERROR,
    SOLVER_STATUS_INFEASIBLE,
    SOLVER_STATUS_OPTIMAL,
    SOLVER_STATUS_TIMEOUT,
    SOLVER_STATUS_UNBOUNDED,
    StrategyProfile,
    TOTAL_SQUAD_SIZE,
)

logger = logging.getLogger(__name__)

__all__ = ["solve_strategy", "_build_player_index"]


# ---------------------------------------------------------------------------
# Pre-flight checks (fail-fast with explicit diagnostics)
# ---------------------------------------------------------------------------


class PreFlightError(ValueError):
    """Raised when the pool is structurally insufficient for the request."""


def _preflight_mantra(pool: list[Player], config: OptimizationConfig) -> None:
    """MANTRA-specific pool checks: eligible_roles present and quota coverable."""
    players_without_roles = [p.player_id for p in pool if not p.eligible_roles]
    if players_without_roles:
        raise PreFlightError(
            f"MANTRA ruleset requires eligible_roles on every player; "
            f"missing for: {players_without_roles[:5]}{'...' if len(players_without_roles) > 5 else ''}"
        )
    quotas = config.mantra_role_quotas or MANTRA_DEFAULT_QUOTAS
    # Count how many players are eligible for each Mantra role.
    coverage: dict[str, int] = defaultdict(int)
    for p in pool:
        for r in p.eligible_roles:
            coverage[r] += 1
    for role, quota in quotas.items():
        if quota > 0 and coverage[role] < quota:
            raise PreFlightError(
                f"MANTRA pool has {coverage[role]} players eligible for role {role!r}, "
                f"required quota is {quota}"
            )


def _preflight(pool: list[Player], config: OptimizationConfig) -> None:
    """Validates pool size and per-role coverage; raises PreFlightError on failure."""
    if not pool:
        raise PreFlightError("Pool is empty")

    by_role: dict[str, list[Player]] = defaultdict(list)
    by_team_pp: dict[str, list[Player]] = defaultdict(list)
    for p in pool:
        by_role[p.role].append(p)
        by_team_pp[p.real_team].append(p)

    if config.ruleset == "MANTRA":
        _preflight_mantra(pool, config)
    else:
        for role, quota in ROLE_QUOTAS.items():
            if len(by_role[role]) < quota:
                raise PreFlightError(
                    f"Pool has {len(by_role[role])} players for role {role!r}, "
                    f"required quota is {quota}"
                )

    distinct_teams = {p.real_team for p in pool}
    if len(distinct_teams) < config.min_distinct_teams:
        raise PreFlightError(
            f"Pool has {len(distinct_teams)} distinct real teams, "
            f"required at least {config.min_distinct_teams} for min_distinct_teams"
        )

    # Capacity check: with max_players_per_team=N and team size k_t, the
    # maximum number of selectable players is Σ min(N, k_t). If this is
    # below TOTAL_SQUAD_SIZE the ILP will be infeasible regardless of
    # other constraints; surface it as a preflight failure with an
    # actionable message.
    capacity = sum(
        min(config.max_players_per_team, len(pids)) for pids in by_team_pp.values()
    )
    if capacity < TOTAL_SQUAD_SIZE:
        raise PreFlightError(
            f"Pool capacity under max_players_per_team={config.max_players_per_team} "
            f"is {capacity}, required at least {TOTAL_SQUAD_SIZE} to fill the squad"
        )

    # Heuristic: cannot satisfy budget if the cheapest possible selection
    # exceeds the budget.
    cheapest_selection_cost = 0.0
    for role, quota in ROLE_QUOTAS.items():
        sorted_by_cost = sorted(by_role[role], key=lambda x: x.cost)
        cheapest_selection_cost += sum(p.cost for p in sorted_by_cost[:quota])
    if cheapest_selection_cost > config.budget:
        raise PreFlightError(
            f"Cheapest possible selection costs {cheapest_selection_cost} credits, "
            f"exceeds budget {config.budget}"
        )


def _build_player_index(pool: list[Player]) -> dict[str, list[str]]:
    """Returns mapping ``real_team -> list[player_id]`` for fast constraint building."""
    by_team: dict[str, list[str]] = defaultdict(list)
    for p in pool:
        by_team[p.real_team].append(p.player_id)
    return dict(by_team)


# ---------------------------------------------------------------------------
# Role constraint builders (Classic vs Mantra)
# ---------------------------------------------------------------------------


def _build_role_constraints_classic(
    prob: pulp.LpProblem,
    x: dict[str, pulp.LpVariable],
    pool: list[Player],
    config: OptimizationConfig,
) -> None:
    """Classic: one binary variable per player, equality constraints on 4 roles."""
    for role, quota in ROLE_QUOTAS.items():
        prob += (
            pulp.lpSum(x[p.player_id] for p in pool if p.role == role) == quota,
            f"quota_{role}",
        )


def _build_role_constraints_mantra(
    prob: pulp.LpProblem,
    x: dict[str, pulp.LpVariable],
    pool: list[Player],
    config: OptimizationConfig,
    x_ir: dict[tuple[str, str], pulp.LpVariable],
) -> None:
    """Mantra: per-(player, role) variables with multi-slot eligibility.

    x_ir[(player_id, role)] == 1 iff player_id fills slot for role.
    select_i = Σ_r x_ir[(i, r)] ≤ 1  (player fills at most one slot)
    x[player_id] == select_i           (bridge to the main x variable for budget/team constraints)
    Σ_i x_ir[(i, r)] == quota[r]       (role quota)
    """
    quotas = config.mantra_role_quotas or MANTRA_DEFAULT_QUOTAS

    for p in pool:
        eligible = p.eligible_roles if p.eligible_roles else frozenset()
        # select_i = Σ_r x_ir[(i, r)]
        select_expr = pulp.lpSum(
            x_ir[(p.player_id, r)] for r in eligible if (p.player_id, r) in x_ir
        )
        # Bridge: x[i] == select_i (so budget/team constraints on x still hold)
        prob += (x[p.player_id] == select_expr, f"select_{p.player_id}")
        # At most one role slot per player
        prob += (select_expr <= 1, f"one_slot_{p.player_id}")

    for role, quota in quotas.items():
        if quota == 0:
            continue
        prob += (
            pulp.lpSum(
                x_ir[(p.player_id, role)]
                for p in pool
                if (p.player_id, role) in x_ir
            )
            == quota,
            f"mantra_quota_{role}",
        )


_ROLE_CONSTRAINT_BUILDERS = {
    "CLASSIC": _build_role_constraints_classic,
}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_strategy(
    pool: list[Player],
    config: OptimizationConfig,
    strategy: StrategyProfile,
    *,
    precomputed_percentiles: dict[str, float] | None = None,
) -> OptimizationResult:
    """Solve the 0/1 selection problem for a single strategy.

    Raises :class:`PreFlightError` if the pool is structurally insufficient
    (handled by the orchestrator to produce a per-strategy ``INFEASIBLE``
    result with diagnostics).

    All other exceptions (solver crash, OOM, etc.) are caught and returned
    as ``OptimizationResult(status=ERROR, ...)``.
    """
    started = time.perf_counter()

    try:
        _preflight(pool, config)
    except PreFlightError as exc:
        logger.warning(
            "preflight_failed strategy=%s reason=%s",
            getattr(strategy, "name", "?"),
            exc,
        )
        return _empty_infeasible_result(
            strategy_name=strategy.name,
            reason=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )

    # Compute effective costs up-front (pure function, decoupled from solver).
    if precomputed_percentiles is None:
        precomputed_percentiles = compute_role_percentile_map(pool)

    try:
        effective_cost: dict[str, float] = {
            p.player_id: estimate_effective_cost(
                player=p,
                role_percentile=precomputed_percentiles[p.player_id],
                num_participants=config.num_participants,
                config=config.inflation_config,
            )
            for p in pool
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("effective_cost_computation_failed strategy=%s", strategy.name)
        return _error_result(strategy.name, str(exc), time.perf_counter() - started)

    # -----------------------------------------------------------------
    # Build PuLP model
    # -----------------------------------------------------------------
    prob = pulp.LpProblem(
        name=f"fantacalcio_{strategy.name}",
        sense=pulp.LpMaximize,
    )

    x = {
        p.player_id: pulp.LpVariable(name=f"x_{p.player_id}", cat=pulp.LpBinary)
        for p in pool
    }

    # MANTRA: per-(player, role) variables for multi-slot eligibility.
    x_ir: dict[tuple[str, str], pulp.LpVariable] = {}
    if config.ruleset == "MANTRA":
        for p in pool:
            for r in p.eligible_roles:
                x_ir[(p.player_id, r)] = pulp.LpVariable(
                    name=f"xir_{p.player_id}_{r}", cat=pulp.LpBinary
                )

    # Objective: maximise Σ role_weight * reliability * risk_adjusted_score * x_i
    # risk_adjusted_score = projected_score - risk_aversion * prediction_std
    # When risk_aversion=0 or prediction_std is None the term collapses to projected_score.
    prob += pulp.lpSum(
        strategy.role_weight[p.role]
        * (p.reliability_weight if p.reliability_weight is not None else 1.0)
        * (
            p.projected_score
            - config.risk_aversion * (p.prediction_std if p.prediction_std is not None else 0.0)
        )
        * x[p.player_id]
        for p in pool
    )

    # -----------------------------------------------------------------
    # Hard constraints (common to every strategy)
    # -----------------------------------------------------------------

    # must_include / exclude: fix binary variables before building anything else.
    for pid in config.must_include:
        if pid in x:
            prob += (x[pid] == 1, f"must_include_{pid}")
    for pid in config.exclude:
        if pid in x:
            prob += (x[pid] == 0, f"exclude_{pid}")

    # Budget on effective costs.
    prob += (
        pulp.lpSum(effective_cost[p.player_id] * x[p.player_id] for p in pool)
        <= float(config.budget),
        "budget_effective",
    )

    # Role quotas — dispatched by ruleset.
    if config.ruleset == "MANTRA":
        _build_role_constraints_mantra(prob, x, pool, config, x_ir)
    else:
        _build_role_constraints_classic(prob, x, pool, config)

    # Max players per real team.
    by_team = _build_player_index(pool)
    for team, pids in by_team.items():
        if len(pids) > 0:
            prob += (
                pulp.lpSum(x[pid] for pid in pids) <= config.max_players_per_team,
                f"max_per_team_{team}",
            )

    # Big teams aggregate cap.
    big_pids = [pid for t, pids in by_team.items() if t in config.big_teams for pid in pids]
    if big_pids:
        prob += (
            pulp.lpSum(x[pid] for pid in big_pids) <= config.big_teams_cap,
            "big_teams_cap",
        )

    # Distinct teams: y_t = 1 iff at least one player from team t is selected.
    # Linearisation: y_t <= Σ_{i in team(t)} x_i
    #                y_t >= x_i    for all i in team(t)
    #                Σ_t y_t >= min_distinct_teams
    y: dict[str, pulp.LpVariable] = {}
    for team, pids in by_team.items():
        y[team] = pulp.LpVariable(name=f"y_{team}", cat=pulp.LpBinary)
        # y_t <= sum_i x_i
        prob += (
            y[team] <= pulp.lpSum(x[pid] for pid in pids),
            f"y_le_team_{team}",
        )
        # y_t >= x_i for each i  (M = max_players_per_team is a valid UB)
        for pid in pids:
            prob += (
                y[team] >= x[pid],
                f"y_ge_{team}_{pid}",
            )
    prob += (
        pulp.lpSum(y[t] for t in y) >= config.min_distinct_teams,
        "min_distinct_teams",
    )

    # Formation feasibility: enforce only the preferred_formation (if set).
    # All modules in config.formations are checked post-hoc in _evaluate_formations
    # and reported in OptimizationResult.formation_feasibility, but are NOT hard
    # constraints — this gives the solver more freedom and makes the output field
    # genuinely informative instead of tautologically True for every module.
    if config.preferred_formation is not None:
        fm = config.preferred_formation
        prob += (
            pulp.lpSum(x[p.player_id] for p in pool if p.role == "D") >= fm.defenders,
            "preferred_fm_D",
        )
        prob += (
            pulp.lpSum(x[p.player_id] for p in pool if p.role == "C") >= fm.midfielders,
            "preferred_fm_C",
        )
        prob += (
            pulp.lpSum(x[p.player_id] for p in pool if p.role == "A") >= fm.forwards,
            "preferred_fm_A",
        )

    # Budget concentration cap: no single player may consume more than
    # max_single_player_budget_share of the total budget (effective cost).
    single_cap = config.max_single_player_budget_share * float(config.budget)
    for p in pool:
        if effective_cost[p.player_id] > single_cap:
            prob += (
                x[p.player_id] == 0,
                f"budget_cap_{p.player_id}",
            )

    # -----------------------------------------------------------------
    # Strategy-specific constraints
    # -----------------------------------------------------------------

    # Min budget share on a set of roles.
    if strategy.min_budget_share_by_roles is not None:
        roles_set, share = strategy.min_budget_share_by_roles
        prob += (
            pulp.lpSum(
                effective_cost[p.player_id] * x[p.player_id]
                for p in pool
                if p.role in roles_set
            )
            >= share * float(config.budget),
            f"min_budget_share_{''.join(sorted(roles_set))}",
        )

    # Top-tier cap: count players with effective_cost >= threshold.
    if strategy.max_top_tier_players is not None and strategy.top_tier_cost_threshold is not None:
        threshold = float(strategy.top_tier_cost_threshold)
        # Pre-filter to candidates; the constraint reduces to a simple sum
        # over the candidate subset.
        top_tier_pids = [
            p.player_id
            for p in pool
            if effective_cost[p.player_id] >= threshold
        ]
        prob += (
            pulp.lpSum(x[pid] for pid in top_tier_pids)
            <= int(strategy.max_top_tier_players),
            "max_top_tier",
        )

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=int(config.solver_timeout_seconds),
    )
    try:
        prob.solve(solver)
    except Exception as exc:  # pragma: no cover - solver crashes are rare
        logger.exception("solver_crash strategy=%s", strategy.name)
        return _error_result(strategy.name, str(exc), time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    status = _map_status(pulp.LpStatus[prob.status], prob)
    logger.info(
        "solver_done strategy=%s status=%s elapsed=%.3fs",
        strategy.name,
        status,
        elapsed,
    )

    if status in (SOLVER_STATUS_INFEASIBLE, SOLVER_STATUS_UNBOUNDED):
        return OptimizationResult(
            strategy_name=strategy.name,
            status=status,  # type: ignore[arg-type]
            squad=[],
            total_nominal_cost=0,
            total_effective_cost=0.0,
            total_projected_score=0.0,
            budget_residual=float(config.budget),
            role_breakdown={r: 0 for r in ROLE_QUOTAS},
            team_breakdown={},
            distinct_teams_count=0,
            big_teams_players_count=0,
            formation_feasibility={fm.label: False for fm in config.formations},
            diagnostics={
                "elapsed_seconds": elapsed,
                "reason": "solver reported no feasible solution",
            },
        )

    selected = [p for p in pool if pulp.value(x[p.player_id]) is not None and pulp.value(x[p.player_id]) > 0.5]
    return _build_result(
        strategy_name=strategy.name,
        selected=selected,
        effective_cost=effective_cost,
        config=config,
        elapsed_seconds=elapsed,
        status=status,
        prob=prob,
    )


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def _extract_mip_gap(prob: pulp.LpProblem) -> float | None:
    """Return MIP gap % if CBC exposes the best bound, else None."""
    try:
        obj = prob.objective.value()
        if obj is None or abs(obj) < 1e-10:
            return None
        # PuLP/CBC stores the best bound in the solver model under different
        # attribute names depending on the version; try the most common ones.
        solver_model = getattr(prob, "solverModel", None)
        if solver_model is None:
            return None
        best_bound = getattr(solver_model, "bestBound", None)
        if best_bound is None:
            best_bound = getattr(solver_model, "ObjBound", None)
        if best_bound is None:
            return None
        return abs(best_bound - obj) / max(abs(obj), 1e-10) * 100.0
    except Exception:  # pragma: no cover - interface not guaranteed across versions
        return None


def _map_status(lp_status: str, prob: pulp.LpProblem) -> str:
    if lp_status == "Optimal":
        return SOLVER_STATUS_OPTIMAL
    if lp_status == "Infeasible":
        return SOLVER_STATUS_INFEASIBLE
    if lp_status == "Unbounded":
        return SOLVER_STATUS_UNBOUNDED
    if lp_status in ("Not Solved", "Undefined", "Aborted"):
        # CBC may abort on timeout; treat as TIMEOUT if we have an incumbent.
        obj = prob.objective.value()
        if obj is not None:
            return SOLVER_STATUS_TIMEOUT
        return SOLVER_STATUS_INFEASIBLE
    return SOLVER_STATUS_ERROR


def _build_result(
    *,
    strategy_name: str,
    selected: list[Player],
    effective_cost: dict[str, float],
    config: OptimizationConfig,
    elapsed_seconds: float,
    status: str,
    prob: pulp.LpProblem | None = None,
) -> OptimizationResult:
    # Enforce squad-size invariant; defensive against solver quirks.
    if len(selected) != TOTAL_SQUAD_SIZE:
        # Pad/truncate is meaningless for an infeasible result; mark as such.
        return OptimizationResult(
            strategy_name=strategy_name,
            status=SOLVER_STATUS_INFEASIBLE,  # type: ignore[arg-type]
            squad=[],
            total_nominal_cost=0,
            total_effective_cost=0.0,
            total_projected_score=0.0,
            budget_residual=float(config.budget),
            role_breakdown={r: 0 for r in ROLE_QUOTAS},
            team_breakdown={},
            distinct_teams_count=0,
            big_teams_players_count=0,
            formation_feasibility={fm.label: False for fm in config.formations},
            diagnostics={
                "elapsed_seconds": elapsed_seconds,
                "reason": (
                    f"solver selected {len(selected)} players, expected {TOTAL_SQUAD_SIZE}"
                ),
            },
        )

    total_nominal = sum(p.cost for p in selected)
    total_effective = sum(effective_cost[p.player_id] for p in selected)
    total_score = sum(p.projected_score for p in selected)
    budget_residual = float(config.budget) - total_effective

    role_breakdown: dict[str, int] = {r: 0 for r in ROLE_QUOTAS}
    team_breakdown: dict[str, int] = {}
    big_teams_players_count = 0
    for p in selected:
        role_breakdown[p.role] = role_breakdown.get(p.role, 0) + 1
        team_breakdown[p.real_team] = team_breakdown.get(p.real_team, 0) + 1
        if p.real_team in config.big_teams:
            big_teams_players_count += 1

    distinct_teams_count = len(team_breakdown)
    formation_feasibility = _evaluate_formations(selected, config)

    return OptimizationResult(
        strategy_name=strategy_name,
        status=status,  # type: ignore[arg-type]
        squad=selected,
        total_nominal_cost=int(total_nominal),
        total_effective_cost=float(total_effective),
        total_projected_score=float(total_score),
        budget_residual=float(budget_residual),
        role_breakdown=role_breakdown,
        team_breakdown=team_breakdown,
        distinct_teams_count=distinct_teams_count,
        big_teams_players_count=big_teams_players_count,
        formation_feasibility=formation_feasibility,
        diagnostics={
            "elapsed_seconds": elapsed_seconds,
            **(
                {"mip_gap_pct": _extract_mip_gap(prob)}
                if status == SOLVER_STATUS_TIMEOUT and prob is not None
                else {}
            ),
        },
    )



def _evaluate_formations(
    selected: list[Player], config: OptimizationConfig
) -> dict[str, bool]:
    counts = {"P": 0, "D": 0, "C": 0, "A": 0}
    for p in selected:
        counts[p.role] = counts.get(p.role, 0) + 1
    out: dict[str, bool] = {}
    for fm in config.formations:
        # Implicit 1-GK feasibility: with 3 P in rosa, a starter exists iff P >= 1.
        out[fm.label] = (
            counts["P"] >= 1
            and counts["D"] >= fm.defenders
            and counts["C"] >= fm.midfielders
            and counts["A"] >= fm.forwards
        )
    return out


def _empty_infeasible_result(
    *, strategy_name: str, reason: str, elapsed_seconds: float
) -> OptimizationResult:
    return OptimizationResult(
        strategy_name=strategy_name,
        status=SOLVER_STATUS_INFEASIBLE,  # type: ignore[arg-type]
        squad=[],
        total_nominal_cost=0,
        total_effective_cost=0.0,
        total_projected_score=0.0,
        budget_residual=0.0,
        role_breakdown={r: 0 for r in ROLE_QUOTAS},
        team_breakdown={},
        distinct_teams_count=0,
        big_teams_players_count=0,
        formation_feasibility={},
        diagnostics={"elapsed_seconds": elapsed_seconds, "reason": reason},
    )


def _error_result(
    strategy_name: str, reason: str, elapsed_seconds: float
) -> OptimizationResult:
    return OptimizationResult(
        strategy_name=strategy_name,
        status=SOLVER_STATUS_ERROR,  # type: ignore[arg-type]
        squad=[],
        total_nominal_cost=0,
        total_effective_cost=0.0,
        total_projected_score=0.0,
        budget_residual=0.0,
        role_breakdown={r: 0 for r in ROLE_QUOTAS},
        team_breakdown={},
        distinct_teams_count=0,
        big_teams_players_count=0,
        formation_feasibility={},
        diagnostics={"elapsed_seconds": elapsed_seconds, "reason": reason},
    )
