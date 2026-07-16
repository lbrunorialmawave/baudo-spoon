"""Optimizer router: ILP-based squad selection exposed via HTTP.

Endpoints
---------

* ``POST /optimize/multi`` — run all (or a subset of) default strategies on
  the configured pool. Returns a per-strategy ``OptimizationResult``.
* ``POST /optimize/single`` — run a single named strategy (faster).
* ``GET  /optimize/strategies`` — list the default strategies exposed by
  the module (Bilanciata, Super-difensiva, Super-offensiva, Mista).

The router is decoupled from the optimizer implementation: the API layer
translates Pydantic schemas to the optimizer's pure data classes and
back. This keeps the optimizer module unaware of FastAPI/HTTP concerns.
"""

from __future__ import annotations

import logging
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..data_repository import DataRepository
from ..deps import get_db, verify_api_key
from ..schemas import (
    DefaultStrategiesResponse,
    MultiStrategyResultSchema,
    OptimizationRequest,
    OptimizationResultSchema,
    SquadPlayerSchema,
    StrategyProfileSchema,
)
from ml.optimizer.inflation import compute_role_percentile_map, estimate_effective_cost
from ml.optimizer.models import (
    Formation,
    InflationConfig,
    OptimizationConfig,
    OptimizationResult,
    Player,
    Role,
    StrategyProfile,
)
from ml.optimizer.optimizer import (
    deduplicate_players,
    optimize_multi_strategy,
    optimize_squad,
)
from ml.optimizer.solver import PreFlightError
from ml.optimizer.strategies import default_strategies

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimize",
    tags=["optimizer"],
    dependencies=[Depends(verify_api_key)],
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_config(req: OptimizationRequest) -> OptimizationConfig:
    """Translate the HTTP request into the optimizer's pure data classes."""
    formations = [
        Formation(
            label=f.label,
            defenders=f.defenders,
            midfielders=f.midfielders,
            forwards=f.forwards,
        )
        for f in req.formations
    ]
    inflation = InflationConfig(
        inflation_percentile_threshold=req.inflation_config.inflation_percentile_threshold,
        max_inflation_multiplier=req.inflation_config.max_inflation_multiplier,
        base_inflation_rate=req.inflation_config.base_inflation_rate,
        baseline_participants=req.inflation_config.baseline_participants,
    )
    return OptimizationConfig(
        budget=req.budget,
        formations=formations,
        num_participants=req.num_participants,
        max_players_per_team=req.max_players_per_team,
        big_teams=frozenset(req.big_teams),
        big_teams_cap=req.big_teams_cap,
        min_distinct_teams=req.min_distinct_teams,
        inflation_config=inflation,
        solver_timeout_seconds=req.solver_timeout_seconds,
    )


def _pool_from_override(req: OptimizationRequest) -> list[Player]:
    """Build ``Player`` objects from a client-supplied pool override."""
    if req.pool_override is None:
        return []
    return [
        Player(
            player_id=p.player_id,
            name=p.name,
            role=cast(Role, p.role),
            real_team=p.real_team,
            cost=p.cost,
            projected_score=p.projected_score,
            reliability_weight=p.reliability_weight,
        )
        for p in req.pool_override
    ]


def _strategy_by_name(name: str) -> StrategyProfile:
    """Look up a default strategy by name; raise 404 on unknown names."""
    for s in default_strategies():
        if s.name == name:
            return s
    valid = ", ".join(s.name for s in default_strategies())
    raise HTTPException(
        status_code=404,
        detail=f"Unknown strategy {name!r}. Valid: {valid}",
    )


def _filter_strategies(
    config: OptimizationConfig,
    names: Optional[list[str]],
) -> list[StrategyProfile]:
    """Pick the requested subset of default strategies (or all)."""
    all_strats = default_strategies()
    if not names:
        return list(all_strats)
    by_name = {s.name: s for s in all_strats}
    selected: list[StrategyProfile] = []
    for n in names:
        if n not in by_name:
            valid = ", ".join(by_name)
            raise HTTPException(
                status_code=404,
                detail=f"Unknown strategy {n!r}. Valid: {valid}",
            )
        selected.append(by_name[n])
    return selected


def _serialize_result(
    result: OptimizationResult,
    effective_cost_lookup: dict[str, float],
) -> OptimizationResultSchema:
    """Translate the optimizer dataclass to the Pydantic response schema."""
    squad = [
        SquadPlayerSchema(
            player_id=p.player_id,
            name=p.name,
            role=p.role,
            real_team=p.real_team,
            cost=p.cost,
            projected_score=p.projected_score,
            effective_cost=effective_cost_lookup.get(p.player_id, float(p.cost)),
        )
        for p in result.squad
    ]
    return OptimizationResultSchema(
        strategy_name=result.strategy_name,
        status=result.status,
        squad=squad,
        total_nominal_cost=result.total_nominal_cost,
        total_effective_cost=result.total_effective_cost,
        total_projected_score=result.total_projected_score,
        budget_residual=result.budget_residual,
        role_breakdown=result.role_breakdown,
        team_breakdown=result.team_breakdown,
        distinct_teams_count=result.distinct_teams_count,
        big_teams_players_count=result.big_teams_players_count,
        formation_feasibility=result.formation_feasibility,
        diagnostics=result.diagnostics,
    )


def _serialize_strategy(s: StrategyProfile) -> StrategyProfileSchema:
    """Translate a ``StrategyProfile`` to the read-only Pydantic schema."""
    if s.min_budget_share_by_roles is None:
        mbsr: Optional[tuple[list[str], float]] = None
    else:
        roles, share = s.min_budget_share_by_roles
        mbsr = (sorted(roles), share)
    return StrategyProfileSchema(
        name=s.name,
        role_weight=s.role_weight,
        min_budget_share_by_roles=mbsr,
        max_top_tier_players=s.max_top_tier_players,
        top_tier_cost_threshold=s.top_tier_cost_threshold,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get(
    "/strategies",
    response_model=DefaultStrategiesResponse,
    summary="Lista delle strategie di ottimizzazione di default",
)
async def list_default_strategies() -> DefaultStrategiesResponse:
    """Espone i profili delle 4 strategie di default (Bilanciata, Super-difensiva, Super-offensiva, Mista)."""
    return DefaultStrategiesResponse(
        strategies=[_serialize_strategy(s) for s in default_strategies()]
    )


@router.post(
    "/multi",
    response_model=MultiStrategyResultSchema,
    summary="Ottimizzazione multi-strategia (fino a 4 varianti di rosa)",
)
async def run_multi_strategy(
    req: OptimizationRequest,
    db: AsyncSession = Depends(get_db),
) -> MultiStrategyResultSchema:
    """Esegue l'ottimizzazione su un sottoinsieme (o tutte) le strategie di default.

    Se ``pool_override`` non è fornito, il pool viene costruito dal DB
    tramite :class:`DataRepository.get_player_pool`, facendo join con le
    predizioni ML di ``results_latest.json``.
    """
    repo = DataRepository(artifacts_dir=settings.artifacts_dir)
    config = _build_config(req)

    if req.pool_override is not None:
        pool = _pool_from_override(req)
    else:
        rows = await repo.get_player_pool(db, season_start=req.season_start, min_qt_a=req.min_qt_a)
        pool = [
            Player(
                player_id=r["player_id"],
                name=r["name"],
                role=cast(Role, r["role"]),
                real_team=r["real_team"],
                cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
            )
            for r in rows
        ]

    pool = deduplicate_players(pool)
    if not pool:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Empty player pool for season_start={req.season_start} "
                f"(min_qt_a={req.min_qt_a}). Check that quotations and ML "
                f"predictions are available."
            ),
        )

    strategies = _filter_strategies(config, req.strategy_names)

    try:
        multi = optimize_multi_strategy(pool, config, strategies=strategies)
    except PreFlightError as exc:
        # Pool is structurally insufficient for *every* strategy.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build an effective-cost lookup so each squad player carries its own
    # value without re-running the inflation model per request.
    percentiles = compute_role_percentile_map(pool)
    effective_lookup: dict[str, float] = {
        p.player_id: estimate_effective_cost(
            player=p,
            role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants,
            config=config.inflation_config,
        )
        for p in pool
    }

    serialized = {
        name: _serialize_result(res, effective_lookup)
        for name, res in multi.results.items()
    }
    return MultiStrategyResultSchema(results=serialized)


@router.post(
    "/single",
    response_model=OptimizationResultSchema,
    summary="Ottimizzazione singola strategia",
)
async def run_single_strategy(
    req: OptimizationRequest,
    strategy_name: str,
    db: AsyncSession = Depends(get_db),
) -> OptimizationResultSchema:
    """Esegue l'ottimizzazione su una singola strategia (più veloce di ``/multi``)."""
    strategy = _strategy_by_name(strategy_name)
    repo = DataRepository(artifacts_dir=settings.artifacts_dir)
    config = _build_config(req)

    if req.pool_override is not None:
        pool = _pool_from_override(req)
    else:
        rows = await repo.get_player_pool(db, season_start=req.season_start, min_qt_a=req.min_qt_a)
        pool = [
            Player(
                player_id=r["player_id"],
                name=r["name"],
                role=cast(Role, r["role"]),
                real_team=r["real_team"],
                cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
            )
            for r in rows
        ]

    pool = deduplicate_players(pool)
    if not pool:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Empty player pool for season_start={req.season_start} "
                f"(min_qt_a={req.min_qt_a})."
            ),
        )

    try:
        result = optimize_squad(pool, config, strategy)
    except PreFlightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    percentiles = compute_role_percentile_map(pool)
    effective_lookup: dict[str, float] = {
        p.player_id: estimate_effective_cost(
            player=p,
            role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants,
            config=config.inflation_config,
        )
        for p in pool
    }
    return _serialize_result(result, effective_lookup)
