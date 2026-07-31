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

import asyncio
import logging
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, require_role
from ..schemas import (
    DefaultStrategiesResponse,
    MultiStrategyResultSchema,
    OptimizationRequest,
    OptimizationResultSchema,
    SquadPlayerSchema,
    StrategyProfileSchema,
)
from ml.optimizer.inflation import compute_role_percentile_map, estimate_effective_cost
from ml.optimizer.team_strength import load_team_strength_scores
from ml.optimizer.win_probability import WinProbabilityConfig, estimate_completion_probability
from ml.optimizer.models import (
    Formation,
    InflationConfig,
    OptimizationConfig,
    OptimizationResult,
    Player,
    Role,
    RulesetType,
    StrategyProfile,
)
from ml.optimizer.optimizer import (
    deduplicate_players,
    optimize_multi_strategy,
    optimize_squad,
)
from ml.optimizer.solver import PreFlightError
from ml.optimizer.strategies import default_strategies
from ml.auction.var import VarEngine

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimize",
    tags=["optimizer"],
    dependencies=[Depends(require_role("member")), Depends(rate_limit)],
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
        team_strength_multiplier=req.inflation_config.team_strength_multiplier,
    )
    preferred_formation = (
        Formation(
            label=req.preferred_formation.label,
            defenders=req.preferred_formation.defenders,
            midfielders=req.preferred_formation.midfielders,
            forwards=req.preferred_formation.forwards,
        )
        if req.preferred_formation is not None
        else None
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
        max_single_player_budget_share=req.max_single_player_budget_share,
        must_include=frozenset(req.must_include),
        exclude=frozenset(req.exclude),
        ruleset=cast(RulesetType, req.ruleset),
        mantra_role_quotas=req.mantra_role_quotas,
        preferred_formation=preferred_formation,
        risk_aversion=req.risk_aversion,
        var_blend=req.var_blend,
        esv_weight=req.esv_weight,
        valuation_mode=req.valuation_mode,
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
            eligible_roles=frozenset(p.eligible_roles),
            prediction_std=p.prediction_std,
            historical_overpay_ratio=p.historical_overpay_ratio,
        )
        for p in req.pool_override
    ]


def _apply_min_start_probability(
    pool: list[Player],
    threshold: Optional[float],
) -> list[Player]:
    """Filter the player pool by ``start_probability >= threshold``.

    Mirrors the contract documented on ``OptimizationRequest.min_start_probability``
    and on ``AuctionConfigSchema.min_start_probability``: players whose
    ``start_probability`` is strictly below the threshold are removed
    BEFORE the ILP solver runs. Players with ``start_probability`` of
    ``None`` (e.g. supplied via ``pool_override`` which does not carry
    the field, or with a DB row missing the value) are kept untouched —
    a missing value is treated as "unknown", not as "low".

    Args:
        pool: Players to filter (not mutated).
        threshold: ``None`` ⇒ no-op (default behavior). Otherwise any
            value in ``[0.0, 1.0]`` is honoured; values outside the range
            are guarded by the Pydantic schema.

    Returns:
        A new list of players; the input is left untouched.
    """
    if threshold is None:
        return pool
    return [
        p for p in pool
        if p.start_probability is None or p.start_probability >= threshold
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


def _custom_strategies(schemas: list[StrategyProfileSchema]) -> list[StrategyProfile]:
    """Convert client-supplied StrategyProfileSchema list → StrategyProfile dataclasses."""
    out: list[StrategyProfile] = []
    for s in schemas:
        mbsr = (
            (frozenset(s.min_budget_share_by_roles[0]), s.min_budget_share_by_roles[1])
            if s.min_budget_share_by_roles is not None
            else None
        )
        try:
            out.append(StrategyProfile(
                name=s.name,
                role_weight=dict(s.role_weight),
                min_budget_share_by_roles=mbsr,
                max_top_tier_players=s.max_top_tier_players,
                top_tier_cost_threshold=s.top_tier_cost_threshold,
            ))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid custom strategy {s.name!r}: {exc}") from exc
    return out


def _serialize_result(
    result: OptimizationResult,
    effective_cost_lookup: dict[str, float],
    win_probability: Optional[float] = None,
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
        win_probability=win_probability,
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
    repo = DataRepository(
        artifacts_dir=settings.artifacts_dir,
        r2_endpoint_url=settings.r2_endpoint_url,
        r2_access_key_id=settings.r2_access_key_id,
        r2_secret_access_key=settings.r2_secret_access_key,
        r2_bucket_name=settings.r2_bucket_name,
    )
    config = _build_config(req)

    if req.pool_override is not None:
        pool = _pool_from_override(req)
    else:
        rows = await repo.get_player_pool(db, season_start=req.season_start, min_qt_a=req.min_qt_a, ruleset=req.ruleset)
        pool = [
            Player(
                player_id=r["player_id"],
                name=r["name"],
                role=cast(Role, r["role"]),
                real_team=r["real_team"],
                cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
                prediction_std=r.get("prediction_std"),
                eligible_roles=frozenset(r.get("eligible_roles") or []),
                start_probability=r.get("start_probability"),
            )
            for r in rows
        ]

    pool = deduplicate_players(pool)
    # Apply client-supplied start_probability pre-filter (mirrors
    # AuctionConfigSchema.min_start_probability semantics). When the
    # threshold is ``None`` (default) this is a no-op.
    pool = _apply_min_start_probability(pool, req.min_start_probability)
    if not pool:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Empty player pool for season_start={req.season_start} "
                f"(min_qt_a={req.min_qt_a}, "
                f"min_start_probability={req.min_start_probability}). "
                "Check that quotations and ML predictions are available, "
                "or lower the threshold."
            ),
        )

    # Enrich pool with VAR/ESV when the config asks for it.
    if config.var_blend > 0 or config.esv_weight > 0:
        engine = VarEngine(
            total_budget=config.budget,
            num_participants=config.num_participants,
            min_start_probability=req.min_start_probability,
            replacement_method=req.replacement_method,
        )
        players_input = [
            {"player_id": p.player_id, "role": p.role,
             "projected_score": p.projected_score, "cost": p.cost,
             "season_value": p.season_value, "start_probability": p.start_probability}
            for p in pool
        ]
        esv_results = engine.evaluate(players_input)
        var_map: dict[str, tuple[float, float]] = {
            e.player_id: (e.var_score, e.esv) for e in esv_results
        }
        from dataclasses import replace
        pool = [
            replace(p, var_score=var_map[p.player_id][0], esv=var_map[p.player_id][1])
            if p.player_id in var_map else p
            for p in pool
        ]

    strategies = (
        _custom_strategies(req.custom_strategies)
        if req.custom_strategies
        else _filter_strategies(req.strategy_names)
    )

    try:
        multi = await asyncio.to_thread(
            optimize_multi_strategy, pool, config, strategies=strategies
        )
    except PreFlightError as exc:
        # Pool is structurally insufficient for *every* strategy.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build an effective-cost lookup so each squad player carries its own
    # value without re-running the inflation model per request.
    percentiles = compute_role_percentile_map(pool)
    known_teams = {p.real_team for p in pool if p.real_team}
    ts_scores = load_team_strength_scores(known_teams=known_teams)
    effective_lookup: dict[str, float] = {
        p.player_id: estimate_effective_cost(
            player=p,
            role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants,
            config=config.inflation_config,
            team_strength_scores=ts_scores,
        )
        for p in pool
    }

    wp_config = WinProbabilityConfig()
    serialized = {
        name: _serialize_result(
            res,
            effective_lookup,
            win_probability=await asyncio.to_thread(
                estimate_completion_probability,
                res.squad, config.budget, wp_config, config.inflation_config, config.num_participants,
            ) if res.squad else None,
        )
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
    strategy_name: str = "",
    db: AsyncSession = Depends(get_db),
) -> OptimizationResultSchema:
    """Esegue l'ottimizzazione su una singola strategia (più veloce di ``/multi``)."""
    if req.custom_strategies:
        if len(req.custom_strategies) != 1:
            raise HTTPException(status_code=422, detail="custom_strategies must have exactly 1 entry for /single")
        strategy = _custom_strategies(req.custom_strategies)[0]
    elif strategy_name:
        strategy = _strategy_by_name(strategy_name)
    else:
        raise HTTPException(status_code=422, detail="Provide strategy_name or custom_strategies")
    repo = DataRepository(
        artifacts_dir=settings.artifacts_dir,
        r2_endpoint_url=settings.r2_endpoint_url,
        r2_access_key_id=settings.r2_access_key_id,
        r2_secret_access_key=settings.r2_secret_access_key,
        r2_bucket_name=settings.r2_bucket_name,
    )
    config = _build_config(req)

    if req.pool_override is not None:
        pool = _pool_from_override(req)
    else:
        rows = await repo.get_player_pool(db, season_start=req.season_start, min_qt_a=req.min_qt_a, ruleset=req.ruleset)
        pool = [
            Player(
                player_id=r["player_id"],
                name=r["name"],
                role=cast(Role, r["role"]),
                real_team=r["real_team"],
                cost=int(r["cost"]),
                projected_score=float(r["projected_score"]),
                prediction_std=r.get("prediction_std"),
                eligible_roles=frozenset(r.get("eligible_roles") or []),
                start_probability=r.get("start_probability"),
            )
            for r in rows
        ]

    pool = deduplicate_players(pool)
    # Apply client-supplied start_probability pre-filter (mirrors
    # AuctionConfigSchema.min_start_probability semantics). When the
    # threshold is ``None`` (default) this is a no-op.
    pool = _apply_min_start_probability(pool, req.min_start_probability)
    if not pool:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Empty player pool for season_start={req.season_start} "
                f"(min_qt_a={req.min_qt_a}, "
                f"min_start_probability={req.min_start_probability})."
            ),
        )

    try:
        result = await asyncio.to_thread(optimize_squad, pool, config, strategy)
    except PreFlightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    percentiles = compute_role_percentile_map(pool)
    known_teams_s = {p.real_team for p in pool if p.real_team}
    ts_scores_s = load_team_strength_scores(known_teams=known_teams_s)
    effective_lookup: dict[str, float] = {
        p.player_id: estimate_effective_cost(
            player=p,
            role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants,
            config=config.inflation_config,
            team_strength_scores=ts_scores_s,
        )
        for p in pool
    }
    wp = await asyncio.to_thread(
        estimate_completion_probability,
        result.squad, config.budget, WinProbabilityConfig(), config.inflation_config, config.num_participants,
    ) if result.squad else None
    return _serialize_result(result, effective_lookup, win_probability=wp)


@router.get(
    "/team-strength",
    summary="Team strength Elo scores (normalized 0–1)",
)
async def get_team_strength() -> dict[str, float]:
    """Return normalized Elo scores for all Serie A clubs."""
    return load_team_strength_scores()
