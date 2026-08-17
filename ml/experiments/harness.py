"""Offline experiment harness (PR5 of the low-sample plan).

The harness runs the **same trainer pipeline** with a matrix of
configuration variants and emits a side-by-side comparison report.  The
report is the single input to the rollout decision in PR8.

The current canonical matrix (plan §36):

* **A — Control**: ``min_minutes=800``, no weighting, no shrinkage, no
  new features.  Reproduces the current production training exactly.
* **B — Weighting**: enable ``enable_limited_sample_training`` and
  ``weighting_strategy=sqrt`` (the recommended default).
* **C — Shrinkage**: enable ``enable_shrinkage`` and reuse the weighted
  cohort from B.
* **D — Recent-role features**: enable PR4 features on top of B.

The harness is intentionally read-only with respect to the production
artifact directory: each variant writes to a sub-folder named
``experiments/<run_id>/<variant>/`` so the canonical ``results_*.json``
remains untouched.

Key invariants (validated by tests):

* The harness does not silently re-implement the trainer — it goes
  through :class:`ml.pipeline.trainer.Trainer` so that the production
  contract is preserved.
* ``A`` must be byte-identical (up to random seed) to the production
  output for the same data hash.  Any divergence is a regression
  candidate and is reported loudly.
* The harness respects ``cfg.test_seasons``; backtest comparisons are
  fair (same train/test cut).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from ..config import MLConfig
from ..pipeline.trainer import Trainer

log = logging.getLogger(__name__)

VARIANT_A: Final[str] = "A_control"
VARIANT_B: Final[str] = "B_weighting"
VARIANT_C: Final[str] = "C_shrinkage"
VARIANT_D: Final[str] = "D_recent_role_features"

# Default matrix; callers can override to drop/add variants.
DEFAULT_VARIANTS: Final[tuple[str, ...]] = (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D)


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    """Configuration overrides for a single experiment variant.

    Attributes:
        name: Identifier (one of ``VARIANT_A`` … ``VARIANT_D``).
        description: Human-readable summary.
        enable_limited_sample_training: Whether LIMITED rows are
            included with reduced weight.
        enable_shrinkage: Whether per-90 shrinkage is applied.
        weighting_strategy: Strategy passed to
            :func:`ml.sample_reliability.weights.compute_sample_weight`.
        enable_recent_role_features: Whether PR4 features are added.
        shrinkage_prior_strength: Pseudo-observation count for the
            shrinkage prior.
    """

    name: str
    description: str
    enable_limited_sample_training: bool
    enable_shrinkage: bool
    weighting_strategy: str
    enable_recent_role_features: bool
    shrinkage_prior_strength: int = 300


def default_variants() -> dict[str, ExperimentVariant]:
    """Return the canonical A/B/C/D matrix as per plan §36."""
    return {
        VARIANT_A: ExperimentVariant(
            name=VARIANT_A,
            description="Control: production behaviour, min_minutes=800, no weighting.",
            enable_limited_sample_training=False,
            enable_shrinkage=False,
            weighting_strategy="sqrt",
            enable_recent_role_features=False,
        ),
        VARIANT_B: ExperimentVariant(
            name=VARIANT_B,
            description="Sample weighting: LIMITED rows with sqrt weights.",
            enable_limited_sample_training=True,
            enable_shrinkage=False,
            weighting_strategy="sqrt",
            enable_recent_role_features=False,
        ),
        VARIANT_C: ExperimentVariant(
            name=VARIANT_C,
            description="Weighting + per-90 shrinkage (default prior=300).",
            enable_limited_sample_training=True,
            enable_shrinkage=True,
            weighting_strategy="sqrt",
            enable_recent_role_features=False,
        ),
        VARIANT_D: ExperimentVariant(
            name=VARIANT_D,
            description="Weighting + shrinkage + recent-role features (PR4).",
            enable_limited_sample_training=True,
            enable_shrinkage=True,
            weighting_strategy="sqrt",
            enable_recent_role_features=True,
        ),
    }


def apply_variant(
    cfg: MLConfig,
    variant: ExperimentVariant,
) -> MLConfig:
    """Return a *copy* of ``cfg`` with the variant overrides applied.

    Pure function: does not mutate the input.  The caller is expected to
    pass the returned config to a fresh :class:`Trainer` instance.

    Fail-closed: after applying the update we **verify** that the boolean
    flags (and related fields) actually landed on the returned config.
    A silent no-op here is exactly the bug that made A/B/C/D bit-identical
    in run ``20260817_065857`` (trainer always dropped at ``min_minutes=800``
    because ``enable_limited_sample_training`` stayed ``False``).
    """
    updates: dict = {
        "enable_limited_sample_training": bool(
            variant.enable_limited_sample_training
        ),
        "enable_shrinkage": bool(variant.enable_shrinkage),
        "weighting_strategy": variant.weighting_strategy,
        "shrinkage_prior_strength": int(variant.shrinkage_prior_strength),
        "enable_recent_role_features": bool(variant.enable_recent_role_features),
        # never auto-enable breakout in experiments
        "enable_breakout_model": False,
    }

    out = cfg.model_copy(update=updates)

    def _mismatches(candidate: MLConfig) -> list[str]:
        bad: list[str] = []
        for key, expected in updates.items():
            actual = getattr(candidate, key)
            if actual != expected:
                bad.append(f"{key}: expected {expected!r}, got {actual!r}")
        return bad

    bad = _mismatches(out)
    if bad:
        # Fallback: bypass Settings env re-binding via model_construct.
        # model_copy on BaseSettings has been observed to drop boolean
        # updates in some runtime/image combinations; constructing from a
        # dumped payload + explicit overrides is deterministic.
        log.warning(
            "apply_variant(%s): model_copy did not stick (%s) — "
            "falling back to model_construct",
            variant.name,
            "; ".join(bad),
        )
        payload = cfg.model_dump()
        payload.update(updates)
        out = MLConfig.model_construct(**payload)
        bad = _mismatches(out)
        if bad:
            raise RuntimeError(
                f"apply_variant({variant.name!r}) failed to apply overrides: "
                + "; ".join(bad)
            )

    log.info(
        "apply_variant(%s): enable_limited_sample_training=%s "
        "enable_shrinkage=%s enable_recent_role_features=%s "
        "weighting_strategy=%s shrinkage_prior_strength=%s",
        variant.name,
        out.enable_limited_sample_training,
        out.enable_shrinkage,
        out.enable_recent_role_features,
        out.weighting_strategy,
        out.shrinkage_prior_strength,
    )
    return out


def run_experiment(
    base_cfg: MLConfig,
    *,
    variants: dict[str, ExperimentVariant] | None = None,
    external_fantavoto_csv=None,
    output_dir: Path | None = None,
) -> dict:
    """Run the experiment matrix and return a comparison report.

    Args:
        base_cfg: Production-equivalent configuration.  Each variant is
            derived from this via :func:`apply_variant`.
        variants: Optional override of the A/B/C/D matrix.
        external_fantavoto_csv: Forwarded to :meth:`Trainer.run`.
        output_dir: Directory to write per-variant artefacts to.
            Defaults to ``<artifacts_dir>/experiments/<run_id>``.

    Returns:
        Comparison report dict (also persisted as ``report.json``).
    """
    variants = variants or default_variants()
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        output_dir or (base_cfg.artifacts_dir / "experiments" / run_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "run_id": run_id,
        "base_config": _cfg_summary(base_cfg),
        "variants": {},
    }

    for name, variant in variants.items():
        log.info("=== Running experiment variant %s ===", name)
        variant_cfg = apply_variant(base_cfg, variant)
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        # Prefer model_copy over copy.copy so BaseSettings fields cannot
        # silently revert when only artifacts_dir is overridden.
        variant_cfg = variant_cfg.model_copy(update={"artifacts_dir": variant_dir})
        if (
            bool(variant_cfg.enable_limited_sample_training)
            != bool(variant.enable_limited_sample_training)
            or bool(variant_cfg.enable_shrinkage) != bool(variant.enable_shrinkage)
            or bool(variant_cfg.enable_recent_role_features)
            != bool(variant.enable_recent_role_features)
        ):
            log.warning(
                "Variant %s flags drifted after artifacts_dir copy — forcing via model_construct",
                name,
            )
            payload = variant_cfg.model_dump()
            payload.update(
                {
                    "enable_limited_sample_training": bool(
                        variant.enable_limited_sample_training
                    ),
                    "enable_shrinkage": bool(variant.enable_shrinkage),
                    "enable_recent_role_features": bool(
                        variant.enable_recent_role_features
                    ),
                    "weighting_strategy": variant.weighting_strategy,
                    "shrinkage_prior_strength": int(variant.shrinkage_prior_strength),
                    "enable_breakout_model": False,
                    "artifacts_dir": variant_dir,
                }
            )
            variant_cfg = MLConfig.model_construct(**payload)
        try:
            trainer = Trainer(variant_cfg)
            output = trainer.run(external_fantavoto_csv=external_fantavoto_csv)
        except Exception as exc:  # noqa: BLE001 — capture and report
            log.exception("Variant %s failed", name)
            report["variants"][name] = {
                "description": variant.description,
                "status": "error",
                "error": repr(exc),
            }
            continue

        summary = _variant_summary(output, variant)
        summary["effective_config"] = _cfg_summary(variant_cfg)
        report["variants"][name] = summary
        log.info(
            "Variant %s done: RMSE=%.4f MAE=%.4f R²=%.4f",
            name,
            summary.get("rmse", float("nan")),
            summary.get("mae", float("nan")),
            summary.get("r2", float("nan")),
        )

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Experiment report written to %s", report_path)
    return report


# ── Internal helpers ────────────────────────────────────────────────────────


def _cfg_summary(cfg: MLConfig) -> dict:
    """Return a JSON-safe subset of the config used in the report."""
    return {
        "min_minutes": cfg.min_minutes,
        "min_minutes_hard": cfg.min_minutes_hard,
        "weighting_strategy": cfg.weighting_strategy,
        "test_seasons": cfg.test_seasons,
        "enable_limited_sample_training": cfg.enable_limited_sample_training,
        "enable_shrinkage": cfg.enable_shrinkage,
        "enable_recent_role_features": cfg.enable_recent_role_features,
        "enable_breakout_model": cfg.enable_breakout_model,
    }


def _variant_summary(output: dict, variant: ExperimentVariant) -> dict:
    """Extract the most important metrics from a trainer run output.

    Includes cohort-stratified error metrics and a phenom-leakage rate
    (plan-limited-cohort-hardening WS4) when prediction rows carry
    ``sample_cohort``.  Absent fields stay ``None`` so older artefacts
    remain reportable.
    """
    role_metrics = output.get("role_metrics", {}) or {}
    out_metrics = (role_metrics.get("outfield") or {}).get(output.get("best_model", "")) or {}
    gk_metrics = (role_metrics.get("gk") or {}).get(output.get("best_model", "")) or {}
    backtest = output.get("backtest", {}) or {}
    cohort = (output.get("sample_reliability") or {}).get("cohort_profile", {}) or {}
    mae_by_cohort, rmse_by_cohort, phenom_leakage = _cohort_stratified_metrics(
        output.get("predictions") or []
    )
    return {
        "description": variant.description,
        "status": "ok",
        "best_model": output.get("best_model"),
        "rmse": out_metrics.get("rmse"),
        "mae": out_metrics.get("mae"),
        "r2": out_metrics.get("r2"),
        "gk_rmse": gk_metrics.get("rmse"),
        "backtest_mean_rmse": backtest.get("mean_rmse"),
        "backtest_mean_mae": backtest.get("mean_mae"),
        "cohort_profile": cohort,
        # WS4 — cohort-aware gate metrics
        "mae_by_cohort": mae_by_cohort,
        "rmse_by_cohort": rmse_by_cohort,
        "phenom_leakage_rate": phenom_leakage,
    }


def _cohort_stratified_metrics(
    predictions: list[dict],
) -> tuple[dict[str, float | None], dict[str, float | None], float | None]:
    """Compute MAE/RMSE per sample_cohort and a phenom-leakage rate.

    Phenom leakage: among players with sample_cohort == LIMITED, the
    fraction that land in the global top-decile of predicted_fantavoto
    (or predicted_fantavoto_display when present).  ``None`` when the
    prediction list lacks the required columns.
    """
    from collections import defaultdict
    import math

    cohorts = ("STANDARD", "LIMITED", "INSUFFICIENT")
    mae_by: dict[str, float | None] = {c: None for c in cohorts}
    rmse_by: dict[str, float | None] = {c: None for c in cohorts}

    if not predictions:
        return mae_by, rmse_by, None

    # Prefer display column for ranking (what Optimizer/Auction see).
    score_key = "predicted_fantavoto"
    if any("predicted_fantavoto_display" in r for r in predictions):
        score_key = "predicted_fantavoto_display"
    target_key = "fantavoto_medio"
    cohort_key = "sample_cohort"

    by_cohort: dict[str, list[tuple[float, float]]] = defaultdict(list)
    scores: list[tuple[str, float]] = []
    for row in predictions:
        cohort = row.get(cohort_key)
        pred = row.get(score_key)
        actual = row.get(target_key)
        if not isinstance(cohort, str) or cohort not in cohorts:
            continue
        if not isinstance(pred, (int, float)):
            continue
        scores.append((cohort, float(pred)))
        if isinstance(actual, (int, float)):
            by_cohort[cohort].append((float(pred), float(actual)))

    for c in cohorts:
        pairs = by_cohort.get(c) or []
        if not pairs:
            continue
        abs_errs = [abs(p - a) for p, a in pairs]
        sq_errs = [(p - a) ** 2 for p, a in pairs]
        mae_by[c] = sum(abs_errs) / len(abs_errs)
        rmse_by[c] = math.sqrt(sum(sq_errs) / len(sq_errs))

    phenom: float | None = None
    limited_scores = [s for c, s in scores if c == "LIMITED"]
    if limited_scores and scores:
        all_pred = sorted(s for _, s in scores)
        # top-decile threshold
        cut_idx = max(0, int(math.ceil(0.9 * len(all_pred))) - 1)
        threshold = all_pred[cut_idx]
        in_top = sum(1 for s in limited_scores if s >= threshold)
        phenom = in_top / len(limited_scores)

    return mae_by, rmse_by, phenom
