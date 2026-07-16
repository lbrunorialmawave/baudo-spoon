"""Multi-strategy orchestrator.

Executes each strategy independently. A failure in one strategy
(``INFEASIBLE``, ``TIMEOUT``, ``ERROR``) never blocks the others; the
``MultiStrategyResult`` always contains an entry per configured strategy.

The 4 strategies are solved **sequentially** (one PuLP process at a time).
This choice is documented: CBC is fast enough on the 500-600 player pool
(typical run < 1s per strategy on standard hardware), and serialising the
solver calls keeps memory bounded, log output linear, and avoids the GIL/
fork complexity of running PuLP in subprocesses.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Sequence

from ml.optimizer.inflation import compute_role_percentile_map
from ml.optimizer.models import (
    MultiStrategyResult,
    OptimizationConfig,
    OptimizationResult,
    Player,
    SOLVER_STATUS_INFEASIBLE,
    StrategyProfile,
)
from ml.optimizer.solver import (
    PreFlightError,
    _empty_infeasible_result,
    _preflight,
    solve_strategy,
)

logger = logging.getLogger(__name__)

__all__ = ["optimize_squad", "optimize_multi_strategy", "deduplicate_players"]


# ---------------------------------------------------------------------------
# Deduplication (handles trasferimenti a stagione in corso + omonimie)
# ---------------------------------------------------------------------------


def deduplicate_players(players: Sequence[Player]) -> list[Player]:
    """Dedup dei giocatori su ``player_id``, mantenendo il record con score più alto.

    Logga un warning per ogni duplicato rimosso e per ogni coppia di
    giocatori con stesso ``name`` ma ``player_id`` diverso (omonimia).
    """
    by_id: dict[str, Player] = {}
    removed_dup: list[str] = []
    for p in players:
        existing = by_id.get(p.player_id)
        if existing is None:
            by_id[p.player_id] = p
        elif p.projected_score > existing.projected_score:
            removed_dup.append(p.player_id)
            by_id[p.player_id] = p
        else:
            removed_dup.append(p.player_id)

    if removed_dup:
        logger.warning(
            "players_deduplicated count=%d unique_kept=%d",
            len(removed_dup),
            len(by_id),
        )

    # Omonimia: stesso name, player_id diverso.
    name_counts = Counter(p.name for p in by_id.values())
    homonyms = {n: c for n, c in name_counts.items() if c > 1}
    if homonyms:
        logger.warning(
            "homonym_players_detected count=%d examples=%s",
            len(homonyms),
            list(homonyms.items())[:5],
        )

    return list(by_id.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_squad(
    pool: Sequence[Player],
    config: OptimizationConfig,
    strategy: StrategyProfile,
) -> OptimizationResult:
    """Run a single strategy on the given pool.

    The pool is deduplicated and validated **before** the solver is invoked.
    Per spec §9.7 (``Pool insufficiente per un ruolo o per min_distinct_teams``),
    a structurally infeasible pool raises :class:`PreFlightError` — this is a
    fail-fast contract: better to surface a clear, actionable error than to
    silently return a placeholder INFEASIBLE for every strategy.

    Parameters
    ----------
    pool:
        Pool di giocatori (verrà deduplicato per ``player_id``).
    config:
        Configurazione completa.
    strategy:
        :class:`StrategyProfile` da applicare.

    Raises
    ------
    PreFlightError
        Se il pool è strutturalmente insufficiente (ruoli mancanti, team
        distinti insufficienti, costo minimo > budget).
    """
    pool_unique = deduplicate_players(pool)
    # Fail-fast: surface structural problems before any solver call.
    _preflight(pool_unique, config)
    percentiles = compute_role_percentile_map(pool_unique)
    return solve_strategy(
        pool=pool_unique,
        config=config,
        strategy=strategy,
        precomputed_percentiles=percentiles,
    )


def optimize_multi_strategy(
    pool: Sequence[Player],
    config: OptimizationConfig,
    *,
    strategies: Sequence[StrategyProfile] | None = None,
) -> MultiStrategyResult:
    """Run every configured strategy and return all results.

    Parameters
    ----------
    pool:
        Pool di giocatori (verrà deduplicato).
    config:
        Configurazione completa; ``config.strategies`` fornisce il default
        se ``strategies`` non è passato.
    strategies:
        Sottoinsieme opzionale di strategie da eseguire. Se ``None``,
        usa ``config.strategies``.

    Returns
    -------
    MultiStrategyResult
        Mappa ``strategy_name -> OptimizationResult``. Contiene sempre
        un'entry per ciascuna strategia richiesta; nessuna strategia può
        far fallire l'intero batch.

    Notes
    -----
    Se il pool è strutturalmente insufficiente, l'eccezione ``PreFlightError``
    viene catturata e **tutte** le strategie richieste ricevono un risultato
    ``INFEASIBLE`` con la stessa motivazione di preflight, mantenendo il
    contratto "4 entry sempre presenti".
    """
    pool_unique = deduplicate_players(pool)
    selected_strategies = list(strategies) if strategies is not None else list(config.strategies)

    if not selected_strategies:
        raise ValueError("No strategies to run")

    # Pre-flight applicato una volta sola (fail-fast a livello di pool).
    # Se fallisce, ogni strategia riceve un INFEASIBLE con la stessa diagnosi.
    try:
        _preflight(pool_unique, config)
    except PreFlightError as exc:
        logger.warning(
            "multi_strategy_preflight_failed reason=%s strategies=%d",
            exc,
            len(selected_strategies),
        )
        results: dict[str, OptimizationResult] = {
            s.name: _empty_infeasible_result(
                strategy_name=s.name,
                reason=str(exc),
                elapsed_seconds=0.0,
            )
            for s in selected_strategies
        }
        return MultiStrategyResult(results=results)

    percentiles = compute_role_percentile_map(pool_unique)
    results = {}
    started = time.perf_counter()

    for strat in selected_strategies:
        logger.info(
            "strategy_start name=%s pool_size=%d budget=%d participants=%d",
            strat.name,
            len(pool_unique),
            config.budget,
            config.num_participants,
        )
        try:
            res = solve_strategy(
                pool=pool_unique,
                config=config,
                strategy=strat,
                precomputed_percentiles=percentiles,
            )
        except PreFlightError as exc:
            # Pre-flight should normally be caught inside solve_strategy,
            # but we double-guard here so a future refactor never breaks
            # the "always return 4 results" contract.
            logger.warning(
                "strategy_preflight_unexpectedly_raised name=%s reason=%s",
                strat.name,
                exc,
            )
            res = _empty_infeasible_result(
                strategy_name=strat.name,
                reason=str(exc),
                elapsed_seconds=0.0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("strategy_unexpected_error name=%s", strat.name)
            res = OptimizationResult(
                strategy_name=strat.name,
                status="ERROR",
                squad=[],
                total_nominal_cost=0,
                total_effective_cost=0.0,
                total_projected_score=0.0,
                budget_residual=float(config.budget),
                role_breakdown={r: 0 for r in ("P", "D", "C", "A")},
                team_breakdown={},
                distinct_teams_count=0,
                big_teams_players_count=0,
                formation_feasibility={},
                diagnostics={"reason": str(exc)},
            )
        results[strat.name] = res
        logger.info(
            "strategy_done name=%s status=%s score=%.2f effective_cost=%.2f",
            strat.name,
            res.status,
            res.total_projected_score,
            res.total_effective_cost,
        )

    total_elapsed = time.perf_counter() - started
    infeasible = [n for n, r in results.items() if r.status == SOLVER_STATUS_INFEASIBLE]
    if infeasible:
        logger.warning(
            "multi_strategy_completed infeasible=%s elapsed=%.3fs",
            infeasible,
            total_elapsed,
        )
    else:
        logger.info("multi_strategy_completed elapsed=%.3fs", total_elapsed)

    return MultiStrategyResult(results=results)
