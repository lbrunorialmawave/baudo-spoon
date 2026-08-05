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
    DiversityMetricsSchema,
    MonteCarloSummarySchema,
    MultiStrategyResultSchema,
    NearOptimalAlternativeSchema,
    OptimizeJobCreateResponse,
    OptimizeJobStatusSchema,
    OptimizationRequest,
    OptimizationResultSchema,
    SquadPlayerSchema,
    StrategyProfileSchema,
)
from ml.optimizer.diagnostics import build_pool_diagnostics, merge_result_diagnostics
from ml.optimizer.diversity import diversify_secondary_strategies, NearOptimalConfig, compute_diversity_metrics, generate_near_optimal_alternatives
from ml.optimizer.inflation import compute_role_percentile_map, estimate_effective_cost
from ml.optimizer.job_store import job_store
from ml.optimizer.monte_carlo_opt import MonteCarloOptConfig, build_simulator_from_pool, run_monte_carlo_opt
from ml.optimizer.residual_integration import build_simulator_preferring_residuals
from ml.optimizer.team_strength import load_team_strength_scores
from ml.optimizer.win_probability import WinProbabilityConfig, estimate_completion_probability
from ml.optimizer.models import (
    Formation, InflationConfig, OptimizationConfig, OptimizationResult,
    Player, Role, RulesetType, StrategyProfile,
)
from ml.optimizer.optimizer import deduplicate_players, optimize_multi_strategy, optimize_squad
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
    if req.pool_override is None:
        return []
    return [
        Player(
            player_id=p.player_id, name=p.name, role=cast(Role, p.role),
            real_team=p.real_team, cost=p.cost, projected_score=p.projected_score,
            reliability_weight=p.reliability_weight, eligible_roles=frozenset(p.eligible_roles),
            prediction_std=p.prediction_std, historical_overpay_ratio=p.historical_overpay_ratio,
            season_value=p.season_value, start_probability=p.start_probability,
        )
        for p in req.pool_override
    ]


def _players_from_pool_rows(rows: list[dict]) -> list[Player]:
    return [
        Player(
            player_id=r["player_id"], name=r["name"], role=cast(Role, r["role"]),
            real_team=r["real_team"], cost=int(r["cost"]), projected_score=float(r["projected_score"]),
            prediction_std=r.get("prediction_std"), eligible_roles=frozenset(r.get("eligible_roles") or []),
            historical_overpay_ratio=r.get("historical_overpay_ratio"),
            season_value=r.get("season_value"), start_probability=r.get("start_probability"),
        )
        for r in rows
    ]


def _mc_config_from_request(req: OptimizationRequest) -> MonteCarloOptConfig | None:
    mc = req.monte_carlo
    if mc is None:
        if not settings.optimizer_mc_default_enabled:
            return None
        n_sim, mode, risk_lambda, min_freq, seed = 200, "saa_frequency", 0.5, 0.0, 42
        timeout = settings.optimizer_saa_timeout_seconds
    else:
        if not mc.enabled:
            return None
        n_sim, mode = mc.n_simulations, mc.mode
        risk_lambda, min_freq, seed = mc.risk_lambda, mc.min_selection_frequency, mc.random_seed
        timeout = mc.timeout_seconds or settings.optimizer_saa_timeout_seconds
    if n_sim > settings.optimizer_max_simulations:
        raise HTTPException(status_code=422, detail=f"n_simulations={n_sim} exceeds max {settings.optimizer_max_simulations}")
    if mode not in ("mean_std", "saa_frequency"):
        raise HTTPException(status_code=422, detail=f"invalid monte_carlo.mode {mode!r}")
    return MonteCarloOptConfig(
        enabled=True, n_simulations=n_sim, mode=mode,  # type: ignore[arg-type]
        risk_lambda=risk_lambda, min_selection_frequency=min_freq, random_seed=seed, timeout_seconds=timeout,
    )


def _near_cfg_from_request(req: OptimizationRequest) -> NearOptimalConfig | None:
    n = req.near_optimal
    if n is None or not n.enabled:
        return None
    return NearOptimalConfig(enabled=True, n_alternatives=n.n_alternatives, exclude_top_m=n.exclude_top_m, max_score_drop_pct=n.max_score_drop_pct)


def _serialize_mc_summary(summary: dict) -> MonteCarloSummarySchema:
    return MonteCarloSummarySchema(**summary)


def _serialize_near_optimal(alts, effective_cost_lookup):
    out = []
    for alt in alts:
        squad = [SquadPlayerSchema(
            player_id=p.player_id, name=p.name, role=p.role, real_team=p.real_team,
            cost=p.cost, projected_score=p.projected_score,
            effective_cost=effective_cost_lookup.get(p.player_id, float(p.cost)),
        ) for p in alt.result.squad]
        out.append(NearOptimalAlternativeSchema(
            excluded_player_ids=list(alt.excluded_player_ids),
            score_delta=round(alt.score_delta, 4), score_delta_pct=round(alt.score_delta_pct, 4),
            squad=squad, total_projected_score=alt.result.total_projected_score, status=alt.result.status,
        ))
    return out


def _with_pool_diagnostics(result, pool, *, extra=None):
    from dataclasses import replace as dc_replace
    return dc_replace(result, diagnostics=merge_result_diagnostics(result, build_pool_diagnostics(pool), extra=extra))


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
    monte_carlo_summary: Optional[MonteCarloSummarySchema] = None,
    near_optimal: Optional[list[NearOptimalAlternativeSchema]] = None,
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
        monte_carlo_summary=monte_carlo_summary,
        near_optimal=near_optimal or [],
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
        pool = _players_from_pool_rows(rows)

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
    mc_cfg = _mc_config_from_request(req)
    near_cfg = _near_cfg_from_request(req)
    top_mc_summary = None
    multi_results: dict[str, OptimizationResult]
    try:
        if mc_cfg is not None:
            sim, _mc_warn, _mc_meta = build_simulator_preferring_residuals(
                pool,
                random_seed=mc_cfg.random_seed,
                artifacts_dir=str(settings.artifacts_dir),
                r2_endpoint_url=settings.r2_endpoint_url,
                r2_access_key_id=settings.r2_access_key_id,
                r2_secret_access_key=settings.r2_secret_access_key,
                r2_bucket_name=settings.r2_bucket_name,
            )
            multi_results, last_summary = {}, None
            for strat in strategies:
                saa = await asyncio.to_thread(run_monte_carlo_opt, pool, config, strat, mc_cfg, sim)
                if saa.representative is None:
                    continue
                multi_results[strat.name] = _with_pool_diagnostics(saa.representative, pool, extra={
                    "mc_wall_time_seconds": saa.wall_time_seconds,
                    "mc_scenarios_completed": saa.scenarios_completed,
                    "mc_mode": saa.mode,
                    "mc_seed": saa.random_seed,
                    "residual_source": _mc_meta.get("residual_source"),
                    "residual_using": _mc_meta.get("using"),
                    "residual_rows": _mc_meta.get("n_rows"),
                    "residual_merged_rows": _mc_meta.get("merged_rows"),
                })
                last_summary = saa.to_summary_dict()
                last_summary["residual_source"] = _mc_meta.get("residual_source")
                last_summary["residual_using"] = _mc_meta.get("using")
                last_summary["residual_rows"] = _mc_meta.get("n_rows")
                last_summary["residual_merged_rows"] = _mc_meta.get("merged_rows")
                if _mc_warn:
                    last_summary.setdefault("warnings", [])
                    for w in _mc_warn:
                        if w not in last_summary["warnings"]:
                            last_summary["warnings"].append(w)
            if last_summary:
                top_mc_summary = _serialize_mc_summary(last_summary)
            if not multi_results:
                raise HTTPException(status_code=400, detail="Monte Carlo produced no squads")
        else:
            multi = await asyncio.to_thread(optimize_multi_strategy, pool, config, strategies=strategies)
            multi_results = {n: _with_pool_diagnostics(r, pool) for n, r in multi.results.items()}
    except PreFlightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    percentiles = compute_role_percentile_map(pool)
    known_teams = {p.real_team for p in pool if p.real_team}
    ts_scores = load_team_strength_scores(known_teams=known_teams)
    effective_lookup = {
        p.player_id: estimate_effective_cost(
            player=p, role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants, config=config.inflation_config, team_strength_scores=ts_scores,
        )
        for p in pool
    }
    # Optional soft diversity: re-solve secondary strategies excluding primary core
    if getattr(req, "diversify_strategies", False) and len(multi_results) > 1:
        multi_results = await asyncio.to_thread(
            diversify_secondary_strategies, pool, config, strategies, multi_results,
        )
        multi_results = {n: _with_pool_diagnostics(r, pool) for n, r in multi_results.items()}

    diversity = compute_diversity_metrics(multi_results)

    wp_config = WinProbabilityConfig()
    serialized = {}
    first_name = next(iter(multi_results))
    for name, res in multi_results.items():
        near_schemas = []
        if near_cfg is not None and name == first_name:
            strat = next(s for s in strategies if s.name == name)
            alts = await asyncio.to_thread(generate_near_optimal_alternatives, pool, config, strat, res, near_cfg)
            near_schemas = _serialize_near_optimal(alts, effective_lookup)
        serialized[name] = _serialize_result(
            res, effective_lookup,
            win_probability=await asyncio.to_thread(
                estimate_completion_probability, res.squad, config.budget, wp_config, config.inflation_config, config.num_participants,
            ) if res.squad else None,
            monte_carlo_summary=top_mc_summary if mc_cfg is not None else None,
            near_optimal=near_schemas,
        )
    return MultiStrategyResultSchema(results=serialized, monte_carlo_summary=top_mc_summary, diversity=DiversityMetricsSchema(**diversity.to_dict()))


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
        pool = _players_from_pool_rows(rows)

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

    mc_cfg = _mc_config_from_request(req)
    near_cfg = _near_cfg_from_request(req)
    mc_summary_schema = None
    try:
        if mc_cfg is not None:
            saa = await asyncio.to_thread(run_monte_carlo_opt, pool, config, strategy, mc_cfg)
            if saa.representative is None:
                raise HTTPException(status_code=400, detail="Monte Carlo produced no squad")
            result = _with_pool_diagnostics(saa.representative, pool, extra={
                "mc_wall_time_seconds": saa.wall_time_seconds, "mc_scenarios_completed": saa.scenarios_completed,
                "mc_mode": saa.mode, "mc_seed": saa.random_seed,
            })
            mc_summary_schema = _serialize_mc_summary(saa.to_summary_dict())
        else:
            result = _with_pool_diagnostics(await asyncio.to_thread(optimize_squad, pool, config, strategy), pool)
    except PreFlightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    percentiles = compute_role_percentile_map(pool)
    known_teams_s = {p.real_team for p in pool if p.real_team}
    ts_scores_s = load_team_strength_scores(known_teams=known_teams_s)
    effective_lookup = {
        p.player_id: estimate_effective_cost(
            player=p, role_percentile=percentiles[p.player_id],
            num_participants=config.num_participants, config=config.inflation_config, team_strength_scores=ts_scores_s,
        )
        for p in pool
    }
    near_schemas = []
    if near_cfg is not None:
        alts = await asyncio.to_thread(generate_near_optimal_alternatives, pool, config, strategy, result, near_cfg)
        near_schemas = _serialize_near_optimal(alts, effective_lookup)
    wp = await asyncio.to_thread(
        estimate_completion_probability, result.squad, config.budget, WinProbabilityConfig(), config.inflation_config, config.num_participants,
    ) if result.squad else None
    return _serialize_result(result, effective_lookup, win_probability=wp, monte_carlo_summary=mc_summary_schema, near_optimal=near_schemas)


@router.get(
    "/team-strength",
    summary="Team strength Elo scores (normalized 0–1)",
)
async def get_team_strength() -> dict[str, float]:
    """Return normalized Elo scores for all Serie A clubs."""
    return load_team_strength_scores()


# ── Async Monte Carlo jobs (Phase 4) ─────────────────────────────────────────

def _run_mc_job_sync(job_id, pool, config, strategy, mc_cfg):
    try:
        job_store.set_running(job_id)
        sim, mc_warn, mc_meta = build_simulator_preferring_residuals(
            pool,
            random_seed=mc_cfg.random_seed,
            artifacts_dir=str(settings.artifacts_dir),
            r2_endpoint_url=settings.r2_endpoint_url,
            r2_access_key_id=settings.r2_access_key_id,
            r2_secret_access_key=settings.r2_secret_access_key,
            r2_bucket_name=settings.r2_bucket_name,
        )
        saa = run_monte_carlo_opt(pool, config, strategy, mc_cfg, sim)
        if saa.representative is None:
            job_store.set_failed(job_id, "Monte Carlo produced no representative squad"); return
        result = _with_pool_diagnostics(saa.representative, pool, extra={
            "mc_wall_time_seconds": saa.wall_time_seconds, "mc_scenarios_completed": saa.scenarios_completed,
            "mc_mode": saa.mode, "mc_seed": saa.random_seed, "async_job": True,
            "residual_source": mc_meta.get("residual_source"),
            "residual_using": mc_meta.get("using"),
            "residual_rows": mc_meta.get("n_rows"),
            "residual_merged_rows": mc_meta.get("merged_rows"),
        })
        percentiles = compute_role_percentile_map(pool)
        known = {p.real_team for p in pool if p.real_team}
        ts = load_team_strength_scores(known_teams=known)
        effective_lookup = {
            p.player_id: estimate_effective_cost(
                player=p, role_percentile=percentiles[p.player_id],
                num_participants=config.num_participants, config=config.inflation_config, team_strength_scores=ts,
            ) for p in pool
        }
        summary = saa.to_summary_dict()
        summary["residual_source"] = mc_meta.get("residual_source")
        summary["residual_using"] = mc_meta.get("using")
        summary["residual_rows"] = mc_meta.get("n_rows")
        summary["residual_merged_rows"] = mc_meta.get("merged_rows")
        if mc_warn:
            summary.setdefault("warnings", [])
            for w in mc_warn:
                if w not in summary["warnings"]:
                    summary["warnings"].append(w)
        schema = _serialize_result(result, effective_lookup, monte_carlo_summary=_serialize_mc_summary(summary))
        job_store.set_completed(job_id, result=schema.model_dump(), monte_carlo_summary=summary)
    except Exception as exc:
        log.exception("async MC job %s failed", job_id)
        job_store.set_failed(job_id, str(exc))


@router.post("/jobs", response_model=OptimizeJobCreateResponse)
async def create_optimize_job(req: OptimizationRequest, strategy_name: str = "Bilanciata", db=Depends(get_db)):
    mc_cfg = _mc_config_from_request(req)
    if mc_cfg is None:
        raise HTTPException(status_code=422, detail="monte_carlo.enabled must be true for async jobs")
    strategy = _strategy_by_name(strategy_name) if not req.custom_strategies else _custom_strategies(req.custom_strategies)[0]
    config = _build_config(req)
    if req.pool_override is not None:
        pool = _pool_from_override(req)
    else:
        repo = DataRepository(
            artifacts_dir=settings.artifacts_dir, r2_endpoint_url=settings.r2_endpoint_url,
            r2_access_key_id=settings.r2_access_key_id, r2_secret_access_key=settings.r2_secret_access_key,
            r2_bucket_name=settings.r2_bucket_name,
        )
        rows = await repo.get_player_pool(db, season_start=req.season_start, min_qt_a=req.min_qt_a, ruleset=req.ruleset)
        pool = _players_from_pool_rows(rows)
    pool = deduplicate_players(pool)
    pool = _apply_min_start_probability(pool, req.min_start_probability)
    if not pool:
        raise HTTPException(status_code=400, detail="Empty player pool")
    job = job_store.create(request_meta={"n_simulations": mc_cfg.n_simulations, "mode": mc_cfg.mode, "strategy": strategy.name})
    asyncio.get_running_loop().run_in_executor(None, _run_mc_job_sync, job.job_id, pool, config, strategy, mc_cfg)
    return OptimizeJobCreateResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=OptimizeJobStatusSchema)
async def get_optimize_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    result_schema = OptimizationResultSchema(**job.result) if job.result is not None else None
    mc_summary = MonteCarloSummarySchema(**job.monte_carlo_summary) if job.monte_carlo_summary is not None else None
    return OptimizeJobStatusSchema(
        job_id=job.job_id, status=job.status, created_at=job.created_at, updated_at=job.updated_at,
        error=job.error, result=result_schema, monte_carlo_summary=mc_summary,
    )
