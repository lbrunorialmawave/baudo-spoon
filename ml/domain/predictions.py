"""Domain contracts for ML predictions.

This module owns the canonical derivation of ``fantapunti_totali`` and
``probabilita_titolarita`` (a.k.a. ``season_value`` and
``start_probability``) from the prediction artefact. Multiple call sites
need this logic:

* :mod:`ml.pipeline.trainer` writes the values into the artefact
  (DataFrame-level, vectorised).
* :class:`api.src.data_repository.DataRepository` reads them back when
  building the optimizer pool (per-record, with fallback to the
  pool's own ``projected_score``).
* :func:`ml.mantra.runner.run_mantra` projects them onto the MANTRA
  artefact (per-record, derived from the prediction record directly).

Two helpers expose this contract at the appropriate granularity:

* :func:`resolve_season_value_fields` — single-record lookup, returns
  ``(season_value, start_probability)`` as ``Optional[float]`` so the
  JSON output round-trips cleanly to ``null``.
* :func:`derive_season_value_columns` — DataFrame-level helper used by
  the training pipeline to write the two columns into the artefact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SHAP_TOLERANCE: float = 1e-4
"""Maximum allowed deviation: |Σ shap_values + base_value - prediction|."""

# Italian Serie A season length used to normalise ``expected_minutes`` into
# a starter-probability estimate. 38 matchdays × 90 minutes.
_MATCHDAYS_PER_SEASON: int = 38
_MINUTES_PER_MATCH: int = 90
_FULL_SEASON_MINUTES: float = float(_MATCHDAYS_PER_SEASON * _MINUTES_PER_MATCH)


def _coerce_non_nan_number(value: Any) -> float | None:
    """Return ``value`` as a non-NaN float, or ``None`` for NaN/missing.

    Booleans are explicitly rejected (``bool`` is a subclass of ``int`` in
    Python and would otherwise sneak through as ``0.0`` / ``1.0``).
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if np.isnan(result):
        return None
    return result


def resolve_season_value_fields(
    prediction: Mapping[str, Any] | None,
    *,
    fallback_predicted_score: float | None = None,
) -> tuple[float | None, float | None]:
    """Extract ``(season_value, start_probability)`` from a prediction record.

    The MANTRA artefact and the optimizer pool both expose these two
    fields alongside ``FP_Mantra`` / ``VR`` / ``projected_score`` without
    blending, reconciling, or taking any position on the still-open
    4-pillar vs. ML-pipeline question (see P1-4): this helper is purely
    informational plumbing.

    Resolution rules (in priority order, mirror the trainer's pipeline):

    ``season_value``
        1. If ``fantapunti_totali`` is a non-null number in the
           prediction record → use it verbatim.
        2. Otherwise, when ``expected_minutes > 0`` and a positive
           predicted score is available (from
           ``predicted_fantavoto``, or the caller-supplied
           ``fallback_predicted_score``), derive
           ``predicted_score × (expected_minutes / 90)``.
        3. Otherwise → ``None``.

    ``start_probability``
        1. If ``probabilita_titolarita`` is a non-null number in the
           prediction record → use it verbatim.
        2. Otherwise, when ``expected_minutes ≥ 0``, derive
           ``clip(expected_minutes / 3420, 0, 1)``.
        3. Otherwise → ``None``.

    Args:
        prediction: A single record from the ``predictions`` array of the
            ML artefact (``results_latest.json``). ``None`` is allowed and
            short-circuits to ``(None, None)``.
        fallback_predicted_score: Used only when the prediction record
            carries no pre-computed ``fantapunti_totali`` *and* no
            ``predicted_fantavoto``. Mirrors the historical
            ``data_repository.get_player_pool`` behaviour, where the
            pool's own ``projected_score`` (with its own fallbacks to
            ``fantavoto_medio`` / ``fvm``) is preferred for the
            derivation so the two figures stay consistent for the
            optimizer.

    Returns:
        ``(season_value, start_probability)``. Each component is
        ``None`` whenever no usable input is available, matching the
        existing ``Fase7`` / ``rischio`` ``None`` pattern on the MANTRA
        artefact.
    """
    if prediction is None:
        return None, None

    # ── season_value ────────────────────────────────────────────────────────
    season_value: float | None = None
    fpt = _coerce_non_nan_number(prediction.get("fantapunti_totali"))
    if fpt is not None:
        season_value = fpt
    else:
        em = _coerce_non_nan_number(prediction.get("expected_minutes"))
        if em is not None and em > 0:
            pf = _coerce_non_nan_number(prediction.get("predicted_fantavoto"))
            score = pf if pf is not None else fallback_predicted_score
            if score is not None and score > 0:
                season_value = float(score) * (em / _MINUTES_PER_MATCH)

    # ── start_probability ───────────────────────────────────────────────────
    start_probability: float | None = None
    pt = _coerce_non_nan_number(prediction.get("probabilita_titolarita"))
    if pt is not None:
        start_probability = pt
    else:
        em = _coerce_non_nan_number(prediction.get("expected_minutes"))
        if em is not None and em >= 0:
            start_probability = min(em / _FULL_SEASON_MINUTES, 1.0)

    return season_value, start_probability


def derive_season_value_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised derivation of the two season-value columns.

    Mutates ``predictions_df`` in place by writing two new columns
    (``fantapunti_totali`` and ``probabilita_titolarita``) and returns
    the same DataFrame for chaining. Kept as a separate entry point
    because the training pipeline operates on a DataFrame (vectorised
    ``np.where`` is the natural fit) while the consumer side operates
    on individual records.

    The output values are:

    * ``fantapunti_totali = predicted_fantavoto × (expected_minutes / 90)``
      when ``expected_minutes > 0``, else ``NaN``.
    * ``probabilita_titolarita = clip(expected_minutes / 3420, 0, 1)``
      when ``expected_minutes > 0``, else ``NaN``.

    NaN is the conventional pandas representation for "no data" here; the
    :func:`resolve_season_value_fields` consumer-side helper translates
    it to ``None`` for JSON serialisation.
    """
    _em = predictions_df["expected_minutes"]
    _pf = predictions_df["predicted_fantavoto"]
    _has_minutes = _em > 0
    predictions_df["fantapunti_totali"] = np.where(
        _has_minutes, _pf * (_em / _MINUTES_PER_MATCH), np.nan
    )
    predictions_df["probabilita_titolarita"] = np.where(
        _has_minutes, (_em / _FULL_SEASON_MINUTES).clip(upper=1.0), np.nan
    )
    return predictions_df


@dataclass(frozen=True)
class PredictionExplanation:
    """Full explanation for a single player prediction.

    Args:
        prediction: The model's point prediction.
        confidence: Calibrated confidence score in [0.0, 1.0].
        variance: Estimated predictive variance (>= 0).
        prediction_interval: (lower, upper) credible interval.
        best_case: Optimistic scenario value.
        worst_case: Pessimistic scenario value.
        top_features: List of (feature_name, shap_value) sorted by |shap_value| desc.
        shap_values: Full dict of feature -> shap contribution.
        base_value: SHAP base value (expected model output over training set).
    """

    prediction: float
    confidence: float
    variance: float
    prediction_interval: tuple[float, float]
    best_case: float
    worst_case: float
    top_features: list[tuple[str, float]]
    shap_values: dict[str, float]
    base_value: float

    def __post_init__(self) -> None:
        lo, hi = self.prediction_interval
        if lo > hi:
            raise ValueError(f"prediction_interval lower bound {lo} > upper bound {hi}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.variance < 0.0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")

    def shap_coherence_error(self) -> float:
        """Return |Σ shap_values + base_value - prediction|.

        Should be < SHAP_TOLERANCE for a well-formed explanation.
        """
        return abs(sum(self.shap_values.values()) + self.base_value - self.prediction)

    def is_shap_coherent(self) -> bool:
        """True when SHAP values are consistent with the prediction."""
        return self.shap_coherence_error() < SHAP_TOLERANCE
