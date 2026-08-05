"""Monte Carlo integration for the ILP squad optimizer."""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

import numpy as np

from ml.optimizer.models import OptimizationConfig, OptimizationResult, Player, StrategyProfile
from ml.optimizer.optimizer import optimize_squad
from ml.simulations.monte_carlo import DEFAULT_STD, MonteCarloSimulator, SimulationResult

log = logging.getLogger(__name__)
MonteCarloMode = Literal["mean_std", "saa_frequency"]
__all__ = [
    "MonteCarloOptConfig", "SAAResult", "build_simulator_from_pool",
    "build_simulator_from_residuals", "run_mean_std", "run_saa_frequency", "run_monte_carlo_opt",
]

@dataclass(frozen=True)
class MonteCarloOptConfig:
    enabled: bool = False
    n_simulations: int = 200
    mode: MonteCarloMode = "saa_frequency"
    risk_lambda: float = 0.5
    min_selection_frequency: float = 0.0
    random_seed: int = 42
    timeout_seconds: float = 0.0
    def __post_init__(self) -> None:
        if not 1 <= self.n_simulations <= 1000:
            raise ValueError(f"n_simulations must be in 1..1000, got {self.n_simulations}")
        if self.risk_lambda < 0:
            raise ValueError("risk_lambda must be >= 0")
        if not 0.0 <= self.min_selection_frequency <= 1.0:
            raise ValueError("min_selection_frequency must be in [0, 1]")
        if self.mode not in ("mean_std", "saa_frequency"):
            raise ValueError(f"unsupported mode {self.mode!r}")

@dataclass
class SAAResult:
    mode: MonteCarloMode
    n_simulations: int
    random_seed: int
    selection_frequency: dict[str, float]
    stability_index: float
    mean_pairwise_jaccard: float
    squad_score_percentiles: dict[str, float]
    representative: OptimizationResult | None
    sampling_methods: dict[str, str] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    scenarios_completed: int = 0
    warnings: list[str] = field(default_factory=list)
    def to_summary_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations, "mode": self.mode, "random_seed": self.random_seed,
            "stability_index": round(self.stability_index, 4),
            "selection_frequency": {k: round(v, 4) for k, v in sorted(self.selection_frequency.items(), key=lambda x: -x[1])},
            "squad_score_percentiles": {k: round(v, 3) for k, v in self.squad_score_percentiles.items()},
            "mean_pairwise_jaccard": round(self.mean_pairwise_jaccard, 4),
            "scenarios_completed": self.scenarios_completed,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "sampling_methods_counts": dict(Counter(self.sampling_methods.values())),
            "warnings": list(self.warnings),
        }

def build_simulator_from_residuals(residuals: list[dict], *, random_seed: int = 42) -> MonteCarloSimulator:
    return MonteCarloSimulator(random_seed=random_seed).fit(residuals)

def build_simulator_from_pool(pool: Sequence[Player], *, random_seed: int = 42, n_synthetic: int = 40):
    warnings: list[str] = []
    residuals: list[dict] = []
    rng = np.random.default_rng(random_seed)
    n_with_std = 0
    for p in pool:
        if p.prediction_std is not None and p.prediction_std > 0:
            n_with_std += 1
            for val in rng.normal(0.0, float(p.prediction_std), size=n_synthetic):
                residuals.append({"player_id": p.player_id, "role": p.role, "residual": float(val)})
    if n_with_std == 0:
        warnings.append(f"No prediction_std; parametric fallback DEFAULT_STD={DEFAULT_STD}")
    sim = MonteCarloSimulator(random_seed=random_seed)
    if residuals:
        sim.fit(residuals)
    return sim, warnings

def _simulate_pool(sim, pool, n_simulations):
    payload = [{"player_id": p.player_id, "predicted_score": p.projected_score, "role": p.role} for p in pool]
    return {r.player_id: r for r in sim.simulate_many(payload, n_simulations=n_simulations)}

def run_mean_std(pool, config, strategy, mc_config, simulator=None):
    t0 = time.perf_counter()
    warnings: list[str] = []
    if simulator is None:
        simulator, warnings = build_simulator_from_pool(pool, random_seed=mc_config.random_seed)
    sim_map = _simulate_pool(simulator, pool, mc_config.n_simulations)
    sampling = {pid: r.sampling_method for pid, r in sim_map.items()}
    adjusted = [replace(p, projected_score=max(0.0, sim_map[p.player_id].mean_score - mc_config.risk_lambda * sim_map[p.player_id].std_score)) for p in pool]
    result = optimize_squad(adjusted, config, strategy)
    return SAAResult(
        mode="mean_std", n_simulations=mc_config.n_simulations, random_seed=mc_config.random_seed,
        selection_frequency={p.player_id: 1.0 for p in result.squad} if result.squad else {},
        stability_index=1.0 if result.squad else 0.0, mean_pairwise_jaccard=1.0 if result.squad else 0.0,
        squad_score_percentiles={"p10": result.total_projected_score, "p50": result.total_projected_score, "p90": result.total_projected_score},
        representative=result, sampling_methods=sampling, wall_time_seconds=time.perf_counter() - t0,
        scenarios_completed=1, warnings=warnings,
    )

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0

def run_saa_frequency(pool, config, strategy, mc_config, simulator=None):
    t0 = time.perf_counter()
    warnings: list[str] = []
    if simulator is None:
        simulator, warnings = build_simulator_from_pool(pool, random_seed=mc_config.random_seed)
    sim_map = _simulate_pool(simulator, pool, mc_config.n_simulations)
    sampling = {pid: r.sampling_method for pid, r in sim_map.items()}
    selection_counts: Counter[str] = Counter()
    scenario_squads: list[set[str]] = []
    scenario_scores: list[float] = []
    completed = 0
    pool_by_id = {p.player_id: p for p in pool}
    # Fase 4.4: warm-start each scenario from the previous one's squad — consecutive
    # scenarios share the same model shape and differ only in score, so the prior
    # incumbent is usually still close to optimal and speeds up CBC materially over
    # a cold start across hundreds of near-identical scenario solves.
    prev_warm_start: dict[str, bool] | None = None
    for k in range(mc_config.n_simulations):
        if mc_config.timeout_seconds > 0 and (time.perf_counter() - t0) > mc_config.timeout_seconds:
            warnings.append(f"SAA stopped early at {k}/{mc_config.n_simulations}"); break
        scenario_pool = [replace(p, projected_score=max(0.0, float(sim_map[p.player_id].simulated_scores[k]))) for p in pool]
        try:
            res = optimize_squad(scenario_pool, config, strategy, warm_start=prev_warm_start)
        except Exception as exc:
            warnings.append(f"scenario {k} failed: {exc}"); continue
        completed += 1
        if res.squad:
            ids = {sp.player_id for sp in res.squad}
            prev_warm_start = {pid: (pid in ids) for pid in pool_by_id}
            scenario_squads.append(ids)
            for pid in ids:
                selection_counts[pid] += 1
            scenario_scores.append(sum(pool_by_id[pid].projected_score for pid in ids if pid in pool_by_id))
    freq = {pid: cnt / completed for pid, cnt in selection_counts.items()} if completed else {}
    mean_jaccard = 0.0
    if len(scenario_squads) >= 2:
        pairs = total_j = 0
        sn = min(len(scenario_squads), 50)
        for i in range(sn):
            for j in range(i + 1, sn):
                total_j += _jaccard(scenario_squads[i], scenario_squads[j]); pairs += 1
        mean_jaccard = total_j / pairs if pairs else 0.0
    elif len(scenario_squads) == 1:
        mean_jaccard = 1.0
    mean_pool = [replace(p, projected_score=max(0.0, sim_map[p.player_id].mean_score)) for p in pool]
    try:
        representative = optimize_squad(mean_pool, config, strategy)
    except Exception as exc:
        warnings.append(f"representative solve failed: {exc}"); representative = None
    stability = float(np.mean([freq.get(sp.player_id, 0.0) for sp in representative.squad])) if representative and representative.squad and completed else 0.0
    percentiles = {}
    if scenario_scores:
        arr = np.asarray(scenario_scores, dtype=float)
        percentiles = {"p10": float(np.percentile(arr, 10)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
    if mc_config.min_selection_frequency > 0.0 and representative and representative.squad and completed:
        fragile = sorted(
            (
                (sp.player_id, freq.get(sp.player_id, 0.0))
                for sp in representative.squad
                if freq.get(sp.player_id, 0.0) < mc_config.min_selection_frequency
            ),
            key=lambda x: x[1],
        )
        if fragile:
            names = ", ".join(f"{pid} ({f:.2f})" for pid, f in fragile[:8])
            more = f" (+{len(fragile) - 8} more)" if len(fragile) > 8 else ""
            warnings.append(
                f"{len(fragile)} representative-squad player(s) below "
                f"min_selection_frequency={mc_config.min_selection_frequency}: {names}{more}"
            )
    return SAAResult(
        mode="saa_frequency", n_simulations=mc_config.n_simulations, random_seed=mc_config.random_seed,
        selection_frequency=freq, stability_index=stability, mean_pairwise_jaccard=mean_jaccard,
        squad_score_percentiles=percentiles, representative=representative, sampling_methods=sampling,
        wall_time_seconds=time.perf_counter() - t0, scenarios_completed=completed, warnings=warnings,
    )

def run_monte_carlo_opt(pool, config, strategy, mc_config, simulator=None):
    if not mc_config.enabled:
        raise ValueError("run_monte_carlo_opt called with monte_carlo.enabled=false")
    if mc_config.mode == "mean_std":
        return run_mean_std(pool, config, strategy, mc_config, simulator)
    return run_saa_frequency(pool, config, strategy, mc_config, simulator)
