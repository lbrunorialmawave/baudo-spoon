"""Output-side reliability labelling and display shrinkage (PR9).

Every other module in this package (``cohort.py``, ``shrinkage.py``,
``weights.py``) operates on the *input* side of the model — classifying
rows and re-weighting/shrinking the features that feed the regressor.
Nothing downstream of the model touches the **predicted** value itself,
so a LIMITED-cohort player who posted 2-3 explosive appearances can
still surface a raw ``predicted_fantavoto`` that reads like a phenom,
even though the sample behind it is too small to trust at face value.

This module closes that gap for the *presentation* layer only:

* :func:`attach_output_reliability` labels each prediction row with its
  sample cohort (``INSUFFICIENT`` / ``LIMITED`` / ``STANDARD``, reusing
  :func:`ml.sample_reliability.cohort.classify_cohort`) and a boolean
  ``ml_values_noisy`` flag the frontend can key a badge off.
* It also derives a ``<predicted_col>_display`` column: the same
  Bayesian shrinkage formula used for per-90 input features
  (:func:`ml.sample_reliability.shrinkage.apply_shrinkage`), applied
  this time to the model's own output, pulled toward the per-role
  median of the STANDARD cohort. For STANDARD rows the effect is
  negligible by construction (minutes >> prior_strength); for
  LIMITED/INSUFFICIENT rows it damps outlier predictions toward a
  believable range.

Design principles (consistent with the rest of the package):

* Pure and deterministic — never mutates the input DataFrame, returns
  a new one.
* The **raw** ``predicted_col`` is left untouched. Backtest metrics,
  training, and any other numeric consumer must keep reading the raw
  column; only the new ``_display`` column is damped. Silently
  overwriting the raw prediction would corrupt RMSE/MAE/R2 reporting.
* Rows can be excluded from prior estimation (e.g. cross-league
  fallback rows already flagged ``is_foreign_fallback``) without being
  excluded from classification or damping — same
  exclude-from-prior-only pattern used in
  :meth:`ml.pipeline.trainer.Trainer._apply_shrinkage`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .cohort import COHORT_STANDARD, classify_cohort
from .shrinkage import apply_shrinkage, estimate_prior_rate

log = logging.getLogger(__name__)

# Minimum STANDARD-cohort rows required (per role group) to estimate a
# display-shrinkage prior from that role alone. Below this, falls back
# to the dataset-wide STANDARD cohort median — mirrors
# ``ml.pipeline.trainer._MIN_STANDARD_ROWS_FOR_PRIOR``.
DEFAULT_MIN_STANDARD_ROWS_FOR_PRIOR: int = 30


def attach_output_reliability(
    df: pd.DataFrame,
    *,
    predicted_col: str,
    minutes_col: str = "mins_played",
    role_col: str | None = "canonical_role",
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
    prior_strength: int = 300,
    min_standard_rows_for_prior: int = DEFAULT_MIN_STANDARD_ROWS_FOR_PRIOR,
    exclude_from_prior_mask: pd.Series | None = None,
    display_suffix: str = "_display",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach ``sample_cohort`` / ``ml_values_noisy`` / display columns.

    Args:
        df: Predictions DataFrame. Must contain *predicted_col* and
            *minutes_col*. Not mutated.
        predicted_col: Name of the raw prediction column (e.g.
            ``"predicted_fantavoto"``). Left untouched in the output.
        minutes_col: Column used both for cohort classification and as
            the shrinkage sample-size proxy.
        role_col: Optional column to group priors by (e.g.
            ``"canonical_role"``). ``None`` or missing → single global
            group.
        min_minutes_hard: Lower bound of the LIMITED cohort — rows
            below this are INSUFFICIENT. Should match
            ``cfg.min_minutes_hard``.
        standard_minutes: Lower bound of the STANDARD cohort. Should
            match ``cfg.min_minutes``.
        prior_strength: Pseudo-minutes the prior contributes; higher
            pulls harder toward the median. Should match
            ``cfg.shrinkage_prior_strength`` unless the caller
            deliberately wants a different display-only strength.
        min_standard_rows_for_prior: Minimum STANDARD rows in a role
            group before its own median is trusted; otherwise falls
            back to the global STANDARD median.
        exclude_from_prior_mask: Optional boolean mask (aligned to
            *df*'s index) of rows that must never contribute to the
            prior (e.g. cross-league fallback rows) — they are still
            classified and damped, just not used to compute the prior.
        display_suffix: Suffix appended to *predicted_col* for the new
            damped column.

    Returns:
        ``(new_df, metadata)`` — *new_df* is a copy of *df* with
        ``sample_cohort``, ``ml_values_noisy``, and
        ``f"{predicted_col}{display_suffix}"`` columns added.
        *metadata* is a JSON-safe dict (cohort counts, priors by role)
        suitable for folding into the ``sample_reliability`` output
        section.
    """
    out = df.copy()

    if predicted_col not in out.columns or minutes_col not in out.columns:
        log.warning(
            "attach_output_reliability: missing '%s' or '%s'; skipping (no-op).",
            predicted_col, minutes_col,
        )
        out["sample_cohort"] = COHORT_STANDARD
        out["ml_values_noisy"] = False
        out[f"{predicted_col}{display_suffix}"] = out.get(predicted_col)
        return out, {"enabled": False, "skipped_reason": "required column missing"}

    minutes = pd.to_numeric(out[minutes_col], errors="coerce")
    cohort = minutes.apply(
        lambda m: classify_cohort(
            m, min_minutes_hard=min_minutes_hard, standard_minutes=standard_minutes,
        )
    )
    out["sample_cohort"] = cohort
    out["ml_values_noisy"] = cohort != COHORT_STANDARD

    exclude = (
        exclude_from_prior_mask.reindex(out.index).fillna(False)
        if exclude_from_prior_mask is not None
        else pd.Series(False, index=out.index)
    )
    standard_mask_global = (cohort == COHORT_STANDARD) & ~exclude

    if role_col is not None and role_col in out.columns:
        role_groups = out.groupby(out[role_col].fillna("UNKNOWN")).groups
    else:
        role_groups = {"ALL": out.index}

    priors: dict[str, float] = {}
    display_col = pd.Series(out[predicted_col].to_numpy(dtype=float, copy=True), index=out.index)
    for role, idx in role_groups.items():
        idx = pd.Index(idx)
        role_standard_mask = standard_mask_global.loc[idx]
        use_global_fallback = int(role_standard_mask.sum()) < min_standard_rows_for_prior
        prior_mask = standard_mask_global if use_global_fallback else (
            out.index.isin(idx) & standard_mask_global
        )
        prior_rate = estimate_prior_rate(
            out.loc[prior_mask, predicted_col],
            minutes=minutes.loc[prior_mask],
            min_minutes=standard_minutes,
        )
        priors[str(role)] = prior_rate
        display_col.loc[idx] = apply_shrinkage(
            out.loc[idx, predicted_col],
            minutes=minutes.loc[idx],
            prior_rate=prior_rate,
            prior_strength=prior_strength,
        )

    out[f"{predicted_col}{display_suffix}"] = display_col

    n_noisy = int(out["ml_values_noisy"].sum())
    log.info(
        "Output reliability attached: %d/%d row(s) flagged noisy "
        "(prior_strength=%d, role_groups=%s).",
        n_noisy, len(out), prior_strength, list(role_groups.keys()),
    )
    return out, {
        "enabled": True,
        "predicted_col": predicted_col,
        "display_col": f"{predicted_col}{display_suffix}",
        "prior_strength": int(prior_strength),
        "min_minutes_hard": int(min_minutes_hard),
        "standard_minutes": int(standard_minutes),
        "n_noisy": n_noisy,
        "n_total": int(len(out)),
        "priors_by_role": priors,
    }
