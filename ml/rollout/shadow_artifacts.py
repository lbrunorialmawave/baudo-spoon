"""Emit shadow-mode comparison artifacts without affecting production decisions.

When the deployment layer sets ``ML_*_CHALLENGER=true`` (SHADOW stage), the
trainer / inference path must still use the legacy production config, but
should produce an observable artifact:

    baseline prediction / decision score
    challenger prediction / decision score
    delta
    cohort
    role
    minutes
    canary status

This module is pure (no DB, no model I/O beyond optional caller-supplied rows).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.auction.decision_score import compute_decision_score
from ml.rollout.config_hash import compute_config_hash
from ml.sample_reliability import (
    classify_cohort,
    continuous_reliability_weight,
    get_reliability_weight,
)


@dataclass(frozen=True, slots=True)
class ShadowRow:
    player_id: str
    role: str
    minutes: float
    cohort: str
    baseline_score: float
    challenger_score: float
    delta: float
    canary: bool = False


def score_row_baseline(
    projected_score: float,
    minutes: float | None,
    *,
    mode: str = "bucket",
) -> float:
    """Legacy/production decision score (bucket mode, weight applied)."""
    rw = get_reliability_weight(minutes=minutes, mode=mode)
    return compute_decision_score(
        projected_score=projected_score,
        reliability_weight=rw,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )


def score_row_challenger(
    projected_score: float,
    minutes: float | None,
    *,
    mode: str = "continuous",
) -> float:
    """Challenger decision score (typically continuous weight)."""
    rw = get_reliability_weight(minutes=minutes, mode=mode)
    return compute_decision_score(
        projected_score=projected_score,
        reliability_weight=rw,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )


def build_shadow_rows(
    players: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str = "bucket",
    challenger_mode: str = "continuous",
    canary_ids: set[str] | None = None,
) -> list[ShadowRow]:
    """Build per-player shadow comparison rows.

    Expected keys on each player mapping:
      player_id (or id), role, minutes (or mins_played), projected_score
    """
    canary_ids = canary_ids or set()
    rows: list[ShadowRow] = []
    for p in players:
        pid = str(p.get("player_id") or p.get("id") or "")
        role = str(p.get("role") or p.get("canonical_role") or "?")
        minutes = p.get("minutes", p.get("mins_played"))
        try:
            minutes_f = float(minutes) if minutes is not None else 0.0
        except (TypeError, ValueError):
            minutes_f = 0.0
        projected = float(p.get("projected_score") or p.get("predicted") or 0.0)
        cohort = str(classify_cohort(minutes_f))
        base = score_row_baseline(projected, minutes_f, mode=baseline_mode)
        chal = score_row_challenger(projected, minutes_f, mode=challenger_mode)
        rows.append(
            ShadowRow(
                player_id=pid,
                role=role,
                minutes=minutes_f,
                cohort=cohort,
                baseline_score=base,
                challenger_score=chal,
                delta=chal - base,
                canary=pid in canary_ids,
            )
        )
    return rows


def write_shadow_artifact(
    path: Path | str,
    players: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str = "bucket",
    challenger_mode: str = "continuous",
    canary_ids: set[str] | None = None,
    meta: Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Write a machine-readable shadow comparison JSON artifact.

    If ``config_snapshot`` is provided, its canonical SHA-256
    (``config_hash``, plan §18) is included in the payload so the
    artifact can be cross-checked against the deployment config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_shadow_rows(
        players,
        baseline_mode=baseline_mode,
        challenger_mode=challenger_mode,
        canary_ids=canary_ids,
    )
    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "baseline_mode": baseline_mode,
        "challenger_mode": challenger_mode,
        "n_rows": len(rows),
        "meta": dict(meta or {}),
        "rows": [asdict(r) for r in rows],
    }
    if config_snapshot is not None:
        payload["config_snapshot"] = dict(config_snapshot)
        payload["config_hash"] = compute_config_hash(config_snapshot)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
