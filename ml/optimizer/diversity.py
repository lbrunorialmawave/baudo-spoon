"""Inter-strategy diversity metrics and near-optimal squad alternatives."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field, replace
from typing import Sequence
from ml.optimizer.models import OptimizationConfig, OptimizationResult, Player, StrategyProfile
from ml.optimizer.optimizer import optimize_squad
log = logging.getLogger(__name__)
__all__ = ["DiversityMetrics", "NearOptimalConfig", "NearOptimalAlternative", "compute_diversity_metrics", "generate_near_optimal_alternatives", "DEFAULT_LOW_DIVERSITY_THRESHOLD"]
DEFAULT_LOW_DIVERSITY_THRESHOLD = 0.85

@dataclass(frozen=True)
class DiversityMetrics:
    mean_pairwise_jaccard: float
    max_pairwise_jaccard: float
    min_pairwise_jaccard: float
    mean_overlap_count: float
    max_overlap_count: int
    low_diversity: bool
    pairwise_jaccard: dict[str, float] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "mean_pairwise_jaccard": round(self.mean_pairwise_jaccard, 4),
            "max_pairwise_jaccard": round(self.max_pairwise_jaccard, 4),
            "min_pairwise_jaccard": round(self.min_pairwise_jaccard, 4),
            "mean_overlap_count": round(self.mean_overlap_count, 2),
            "max_overlap_count": self.max_overlap_count,
            "low_diversity": self.low_diversity,
            "pairwise_jaccard": {k: round(v, 4) for k, v in self.pairwise_jaccard.items()},
        }

@dataclass(frozen=True)
class NearOptimalConfig:
    enabled: bool = False
    n_alternatives: int = 3
    exclude_top_m: int = 2
    max_score_drop_pct: float = 0.15
    def __post_init__(self) -> None:
        if self.n_alternatives < 1 or self.exclude_top_m < 1:
            raise ValueError("n_alternatives and exclude_top_m must be >= 1")
        if not 0.0 <= self.max_score_drop_pct <= 1.0:
            raise ValueError("max_score_drop_pct must be in [0, 1]")

@dataclass(frozen=True)
class NearOptimalAlternative:
    excluded_player_ids: tuple[str, ...]
    score_delta: float
    score_delta_pct: float
    result: OptimizationResult

def _jaccard(a, b):
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0

def compute_diversity_metrics(results, *, low_diversity_threshold=DEFAULT_LOW_DIVERSITY_THRESHOLD):
    squads = {n: {p.player_id for p in r.squad} for n, r in results.items() if r.squad}
    names = sorted(squads)
    if len(names) < 2:
        s = 1.0 if names else 0.0
        return DiversityMetrics(s, s, s, 0.0, 0, False, {})
    jaccards, overlaps, pairwise = [], [], {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ja = _jaccard(squads[a], squads[b])
            ov = len(squads[a] & squads[b])
            jaccards.append(ja); overlaps.append(ov); pairwise[f"{a}|{b}"] = ja
    mean_j = sum(jaccards) / len(jaccards)
    return DiversityMetrics(mean_j, max(jaccards), min(jaccards), sum(overlaps)/len(overlaps), max(overlaps), mean_j > low_diversity_threshold, pairwise)

def generate_near_optimal_alternatives(pool, config, strategy, reference, near_cfg):
    if not near_cfg.enabled or not reference.squad:
        return []
    ranked = sorted(reference.squad, key=lambda p: p.projected_score, reverse=True)
    base = reference.total_projected_score
    out = []
    max_start = max(1, len(ranked) - near_cfg.exclude_top_m + 1)
    for start in range(min(near_cfg.n_alternatives, max_start)):
        excluded = ranked[start:start + near_cfg.exclude_top_m]
        if len(excluded) < near_cfg.exclude_top_m and start > 0:
            break
        exclude_ids = {p.player_id for p in excluded}
        new_ex = (set(config.exclude) | exclude_ids) - set(config.must_include)
        try:
            alt = optimize_squad(list(pool), replace(config, exclude=frozenset(new_ex)), strategy)
        except Exception as exc:
            log.warning("near_optimal failed: %s", exc); continue
        if not alt.squad:
            continue
        delta = alt.total_projected_score - base
        pct = delta / base if base else 0.0
        if pct < -near_cfg.max_score_drop_pct:
            continue
        out.append(NearOptimalAlternative(tuple(sorted(exclude_ids)), delta, pct, alt))
    return out
