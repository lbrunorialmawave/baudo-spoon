"""Sample reliability & cohort classification for low-sample player modeling.

This package implements the foundational layer described in ``plan.md``
(Low-Sample Player & Breakout Modeling), specifically:

* **PR1** — ``SampleReliability`` dataclass and cohort classification
  (STANDARD / LIMITED / INSUFFICIENT).
* **PR2** — ``compute_sample_weight`` central function with multiple
  weighting strategies.
* **PR3** — Bayesian shrinkage abstraction for per-90 rate stabilisation
  (kept in a separate module to preserve single-responsibility).
* **PR9** — Output-side reliability labelling (``sample_cohort`` /
  ``ml_values_noisy``) and display-only shrinkage of the model's own
  predictions, so a LIMITED-cohort hot streak isn't presented at face
  value (see ``output_reliability.py``).

Design principles (enforced by the plan):
* Minutes describe the **reliability of the sample**, not the **ability**
  of the player — see :func:`classify_cohort` and the surrounding docstring.
* The historical ``min_minutes = 800`` threshold is preserved as the
  high-confidence cohort boundary.  This module does not silently lower
  the production cutoff.
* The default behaviour of the trainer must remain unchanged when the
  low-sample feature flags are disabled (no-op contract).
* Foreign-fallback rows (``is_foreign_fallback = True``) remain
  **inference-only** by the main plan; this module never promotes them
  to training data.

Changelog note (limited-cohort hardening, 2026-08):
* Continuous decision-layer reliability weight added
  (:func:`continuous_reliability_weight` / :func:`get_reliability_weight`).
  The legacy :data:`RELIABILITY_WEIGHT_BY_COHORT` step function is retained
  as the default (mode="bucket") for bit-identical behaviour; switch to
  mode="continuous" via config once the canary gate passes.
* Historical note: PR2 (weighting) was promoted without PR3 (shrinkage)
  in an earlier rollout — see plan-limited-cohort-hardening.md.
"""

from .cohort import (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
    DEFAULT_RELIABILITY_FLOOR,
    RELIABILITY_WEIGHT_BY_COHORT,
    SAMPLE_COHORTS,
    Cohort,
    SampleReliability,
    classify_cohort,
    continuous_reliability_weight,
    get_reliability_weight,
    profile_dataset,
)
from .output_reliability import (
    DEFAULT_MIN_STANDARD_ROWS_FOR_PRIOR,
    attach_output_reliability,
)
from .shrinkage import (
    DEFAULT_PRIOR_STRENGTH,
    apply_shrinkage,
    estimate_prior_rate,
)
from .weights import (
    STRATEGY_BUCKETED,
    STRATEGY_CONSTANT,
    STRATEGY_LINEAR,
    STRATEGY_SQRT,
    WeightingStrategy,
    compute_sample_weight,
)

__all__ = [
    "COHORT_INSUFFICIENT",
    "COHORT_LIMITED",
    "COHORT_STANDARD",
    "SAMPLE_COHORTS",
    "Cohort",
    "SampleReliability",
    "classify_cohort",
    "profile_dataset",
    "RELIABILITY_WEIGHT_BY_COHORT",
    "DEFAULT_RELIABILITY_FLOOR",
    "continuous_reliability_weight",
    "get_reliability_weight",
    "WeightingStrategy",
    "STRATEGY_CONSTANT",
    "STRATEGY_LINEAR",
    "STRATEGY_SQRT",
    "STRATEGY_BUCKETED",
    "compute_sample_weight",
    "DEFAULT_PRIOR_STRENGTH",
    "apply_shrinkage",
    "estimate_prior_rate",
    "attach_output_reliability",
    "DEFAULT_MIN_STANDARD_ROWS_FOR_PRIOR",
]
