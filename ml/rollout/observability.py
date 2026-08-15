"""Production observability counters for limited-cohort hardening (WS12).

Aggregate, PII-free metrics suitable for logs / dashboards.
Separate raw model / display / decision scores in diagnostic payloads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class CohortObservability:
    """Snapshot of limited-cohort production metrics (no PII)."""

    limited_players_count: int = 0
    standard_players_count: int = 0
    insufficient_players_count: int = 0
    limited_top_decile_count: int = 0
    limited_overrepresentation: float | None = None
    mean_reliability_weight: float | None = None
    mean_minutes_limited: float | None = None
    auction_reliability_enabled: bool | None = None
    optimizer_reliability_enabled: bool | None = None
    rollout_stage: str | None = None
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cohort_observability(
    players: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "projected_score",
    cohort_key: str = "sample_cohort",
    minutes_key: str = "minutes",
    weight_key: str = "reliability_weight",
    top_decile_fraction: float = 0.10,
    auction_reliability_enabled: bool | None = None,
    optimizer_reliability_enabled: bool | None = None,
    rollout_stage: str | None = None,
) -> CohortObservability:
    """Compute aggregate metrics from a player pool (dicts)."""
    cohorts = Counter()
    limited_minutes: list[float] = []
    weights: list[float] = []
    scored: list[tuple[float, str]] = []

    for p in players:
        cohort = str(p.get(cohort_key) or p.get("cohort") or "UNKNOWN").upper()
        cohorts[cohort] += 1
        rw = p.get(weight_key)
        if isinstance(rw, (int, float)):
            weights.append(float(rw))
        mins = p.get(minutes_key, p.get("mins_played"))
        if cohort == "LIMITED" and isinstance(mins, (int, float)):
            limited_minutes.append(float(mins))
        score = p.get(score_key, p.get("decision_score"))
        if isinstance(score, (int, float)):
            scored.append((float(score), cohort))

    limited_top = 0
    overrep: float | None = None
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(len(scored) * top_decile_fraction))
        top = scored[:k]
        limited_top = sum(1 for _, c in top if c == "LIMITED")
        pool_limited = cohorts.get("LIMITED", 0)
        if len(scored) > 0 and pool_limited > 0:
            share_top = limited_top / len(top)
            share_pool = pool_limited / len(scored)
            overrep = share_top / share_pool if share_pool > 0 else None

    return CohortObservability(
        limited_players_count=cohorts.get("LIMITED", 0),
        standard_players_count=cohorts.get("STANDARD", 0),
        insufficient_players_count=cohorts.get("INSUFFICIENT", 0),
        limited_top_decile_count=limited_top,
        limited_overrepresentation=overrep,
        mean_reliability_weight=(sum(weights) / len(weights)) if weights else None,
        mean_minutes_limited=(
            sum(limited_minutes) / len(limited_minutes) if limited_minutes else None
        ),
        auction_reliability_enabled=auction_reliability_enabled,
        optimizer_reliability_enabled=optimizer_reliability_enabled,
        rollout_stage=rollout_stage,
    )


def diagnostic_score_layers(
    *,
    raw_model: float | None,
    display: float | None,
    decision: float | None,
) -> dict[str, float | None]:
    """Separate raw / display / decision for diagnostic logs (no PII)."""
    return {
        "raw_model": raw_model,
        "display": display,
        "decision": decision,
    }
