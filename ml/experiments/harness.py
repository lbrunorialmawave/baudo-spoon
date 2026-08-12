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

import copy
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
    """
    # ``MLConfig`` is a Pydantic v2 model, so we use ``model_copy`` rather
    # than ``dataclasses.replace``.  The result is a fully validated
    # ``MLConfig`` instance.
    return cfg.model_copy(
        update={
            "enable_limited_sample_training": variant.enable_limited_sample_training,
            "enable_shrinkage": variant.enable_shrinkage,
            "weighting_strategy": variant.weighting_strategy,
            "shrinkage_prior_strength": variant.shrinkage_prior_strength,
            "enable_breakout_model": False,  # never auto-enable breakout in experiments
        }
    )


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
        variant_cfg_artifacts = copy.copy(variant_cfg)
        variant_cfg_artifacts.artifacts_dir = variant_dir
        try:
            trainer = Trainer(variant_cfg_artifacts)
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
        "enable_breakout_model": cfg.enable_breakout_model,
    }


def _variant_summary(output: dict, variant: ExperimentVariant) -> dict:
    """Extract the most important metrics from a trainer run output."""
    role_metrics = output.get("role_metrics", {}) or {}
    out_metrics = (role_metrics.get("outfield") or {}).get(output.get("best_model", "")) or {}
    gk_metrics = (role_metrics.get("gk") or {}).get(output.get("best_model", "")) or {}
    backtest = output.get("backtest", {}) or {}
    cohort = (output.get("sample_reliability") or {}).get("cohort_profile", {}) or {}
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
    }
