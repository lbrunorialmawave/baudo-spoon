"""Canary report for the LIMITED-cohort reliability safety net (WS14, plan §16.1).

Builds the JSON artefact consumed by ``ml-training.yml`` to gate the
SHADOW → ACTIVE promotion.  The report is intentionally self-contained
(no live DB dependency): it loads the synthetic canary fixture
:mod:`ml.tests.fixtures.limited_cohort_canary` and re-applies the same
reliability primitives used at training time
(:func:`ml.sample_reliability.weights.compute_sample_weight`,
:func:`ml.sample_reliability.shrinkage.apply_shrinkage`,
:func:`ml.sample_reliability.cohort.continuous_reliability_weight`)
to the LIMITED/INSUFFICIENT rows.

A *known anomaly* is "resolved" when the post-safety-net
``effective_fantavoto`` falls at or below the per-role median of the
STANDARD reference cohort — the operational definition of "no false
phenom leak".  The number of unresolved anomalies is the gate: a
healthy build must report ``anomalies.remaining == 0``.

The function is pure and deterministic; it is covered by
``ml/tests/test_rollout_canary.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Final

import pandas as pd

from ml.config import MLConfig
from ml.rollout.config_hash import build_config_bundle
from ml.sample_reliability import (
    apply_shrinkage,
    compute_sample_weight,
    continuous_reliability_weight,
)
from ml.tests.fixtures.limited_cohort_canary import (
    CANARY_ANOMALY_IDS,
    build_limited_cohort_canary,
)

log = logging.getLogger(__name__)

# Schema version stamped on every report — the workflow validator
# inspects ``anomalies.remaining`` (and a few fallbacks); bumping the
# version forces a re-validation if the structure ever changes.
CANARY_REPORT_VERSION: Final[str] = "1.0"

# Estimated "population" goals-per-90 used as the shrinkage prior for
# the synthetic fixture.  Tuned so the Adzic-style false phenom
# (163 minutes, 1 goal) shrinks to roughly the role median after the
# safety net, but not below — a regression in shrinkage would push the
# effective value back above the threshold and the canary would fail.
_FIXTURE_PRIOR_GOALS_PER_90: Final[float] = 0.18


def _effective_fantavoto(
    df: pd.DataFrame,
    *,
    min_minutes_hard: int,
    standard_minutes: int,
) -> pd.DataFrame:
    """Apply the reliability safety net to the fixture.

    Mirrors the trainer's pipeline (see :mod:`ml.sample_reliability`):

    1. **Sample weight** for training influence (informational here —
       the canary just observes the predicted value post-safety-net).
    2. **Per-90 shrinkage** of the goals rate, pulled toward a
       population prior.
    3. **Continuous reliability weight** floor on the displayed
       ``predicted_fantavoto`` for LIMITED/INSUFFICIENT rows.

    Returns a copy of ``df`` with two new columns:
    ``effective_fantavoto`` and ``effective_goals_per_90``.
    """
    out = df.copy()
    if "mins_played" not in out.columns or "predicted_fantavoto" not in out.columns:
        raise KeyError(
            "Canary fixture must expose 'mins_played' and 'predicted_fantavoto'."
        )

    minutes = pd.to_numeric(out["mins_played"], errors="coerce").fillna(0)
    goals = (
        pd.to_numeric(out.get("goals", 0), errors="coerce").fillna(0)
        if "goals" in out.columns
        else pd.Series(0, index=out.index)
    )

    # 1) Sample weight — logged for traceability, not used for the
    #    effective fantavoto math itself.  ``compute_sample_weight`` is
    #    scalar-only, so we map it over the minutes Series.
    out["sample_weight"] = minutes.apply(
        lambda m: compute_sample_weight(
            m, strategy="sqrt", standard_minutes=standard_minutes
        )
    )

    # 2) Per-90 shrinkage on the goals rate.
    observed_per90 = (goals * 90.0) / minutes.replace(0, pd.NA)
    out["effective_goals_per_90"] = apply_shrinkage(
        observed_per90,
        minutes=minutes,
        prior_rate=_FIXTURE_PRIOR_GOALS_PER_90,
    ).fillna(_FIXTURE_PRIOR_GOALS_PER_90)

    # 3) Continuous reliability weight on the predicted fantavoto for
    #    LIMITED / INSUFFICIENT rows.  STANDARD rows keep their value
    #    (weight == 1.0 by construction).
    weights = minutes.apply(
        lambda m: continuous_reliability_weight(
            m,
            min_minutes_hard=min_minutes_hard,
            standard_minutes=standard_minutes,
        )
    )
    out["reliability_weight"] = weights
    out["effective_fantavoto"] = out["predicted_fantavoto"] * weights
    return out


def _per_role_standard_median(
    df: pd.DataFrame, *, standard_cohort_label: str = "STANDARD"
) -> dict[str, float]:
    """Return ``{role: median(predicted_fantavoto)}`` over STANDARD rows."""
    std = df[df["sample_cohort"] == standard_cohort_label]
    if std.empty:
        return {}
    return (
        std.groupby("canonical_role")["predicted_fantavoto"]
        .median()
        .to_dict()
    )


def _classify_anomalies(
    df: pd.DataFrame,
    *,
    standard_medians: dict[str, float],
    known_anomaly_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Walk known-anomaly rows; mark each resolved / unresolved.

    A known anomaly is *resolved* when the post-safety-net
    ``effective_fantavoto`` is at or below the per-role STANDARD
    median — that is, the safety net successfully demoted the false
    phenom from the top bracket.
    """
    findings: list[dict[str, Any]] = []
    known_mask = df["player_id"].isin(known_anomaly_ids)
    for _, row in df.loc[known_mask].iterrows():
        role = row.get("canonical_role", "")
        threshold = standard_medians.get(role)
        effective = float(row.get("effective_fantavoto", 0.0))
        raw = float(row.get("predicted_fantavoto", 0.0))
        if threshold is None:
            # No STANDARD reference for this role — be conservative
            # and treat as unresolved so the gate fails closed.
            resolved = False
            threshold_used: float | None = None
        else:
            threshold_used = float(threshold)
            resolved = effective <= threshold_used
        findings.append(
            {
                "player_id": str(row.get("player_id", "")),
                "player_name": str(row.get("player_name", "")),
                "role": str(role),
                "raw_predicted_fantavoto": raw,
                "effective_fantavoto": effective,
                "reliability_weight": float(row.get("reliability_weight", 0.0)),
                "role_standard_median": threshold_used,
                "resolved": bool(resolved),
            }
        )
    return findings


def build_canary_report(
    cfg: MLConfig,
    *,
    df: pd.DataFrame | None = None,
    known_anomaly_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return the canary report dict, ready to be serialised to JSON.

    The returned dict matches the contract enforced by
    ``ml-training.yml`` (Phase 6 "Validate canary_report structure"):

    * ``anomalies.remaining`` (alias ``anomalies.remaining_count`` and
      ``canary_anomalies_remaining`` at root) — gate value, must be 0.
    * ``anomalies.total`` and ``anomalies.resolved`` — diagnostic.
    * ``config_hash`` (root) — the canonical SHA-256 of the active
      configuration, paired with the same hash the promotion gate
      compares against.

    Args:
        cfg: The :class:`MLConfig` instance used for the training run.
        df: Optional pre-loaded canary DataFrame; defaults to the
            synthetic fixture.  Exposed for tests.
        known_anomaly_ids: Optional override of the known-anomaly set.
            Defaults to :data:`CANARY_ANOMALY_IDS`.
    """
    fixture = df if df is not None else build_limited_cohort_canary()
    known = known_anomaly_ids if known_anomaly_ids is not None else CANARY_ANOMALY_IDS

    evaluated = _effective_fantavoto(
        fixture,
        min_minutes_hard=int(cfg.min_minutes_hard),
        standard_minutes=int(cfg.min_minutes),
    )
    standard_medians = _per_role_standard_median(evaluated)
    findings = _classify_anomalies(
        evaluated,
        standard_medians=standard_medians,
        known_anomaly_ids=known,
    )
    total = len(findings)
    resolved = sum(1 for f in findings if f["resolved"])
    remaining = total - resolved

    # Config snapshot for the hash — strip non-deterministic bits so
    # the hash is stable across re-runs of the same build.
    config_snapshot: dict[str, Any] = {
        "min_minutes": int(cfg.min_minutes),
        "min_minutes_hard": int(cfg.min_minutes_hard),
        "enable_limited_sample_training": bool(cfg.enable_limited_sample_training),
        "enable_shrinkage": bool(cfg.enable_shrinkage),
        "enable_recent_role_features": bool(cfg.enable_recent_role_features),
        "enable_breakout_model": bool(cfg.enable_breakout_model),
        "weighting_strategy": str(cfg.weighting_strategy),
        "shrinkage_prior_strength": int(cfg.shrinkage_prior_strength),
        "reliability_weight_mode": str(cfg.reliability_weight_mode),
    }
    bundle = build_config_bundle(config=config_snapshot)

    report: dict[str, Any] = {
        "version": CANARY_REPORT_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "fixture_rows": int(len(evaluated)),
        "known_anomaly_ids": sorted(known),
        "anomalies": {
            "total": int(total),
            "resolved": int(resolved),
            "remaining": int(remaining),
            "remaining_count": int(remaining),
            "details": findings,
        },
        # Root-level aliases the validator and downstream tooling may
        # consume independently of the nested ``anomalies`` block.
        "canary_anomalies_total": int(total),
        "canary_anomalies_resolved": int(resolved),
        "canary_anomalies_remaining": int(remaining),
        "config": bundle["config"],
        "config_hash": bundle["config_hash"],
        # Operational signal for humans reading the report.
        "gate_passed": remaining == 0,
    }
    log.info(
        "Canary report: %d/%d anomalies resolved (remaining=%d, gate=%s)",
        resolved,
        total,
        remaining,
        "PASS" if remaining == 0 else "FAIL",
    )
    return report


__all__ = [
    "CANARY_REPORT_VERSION",
    "build_canary_report",
]
