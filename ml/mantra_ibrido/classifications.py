"""Hybrid classifications — labels computed from combined MANTRA+ML scores.

Labels are **not** mutually exclusive: a single player may carry multiple
labels (e.g. ``ML_Confirmed`` + ``Best_Value``).

Output format
-------------
``{ "ML_Confirmed": ["Player A", ...], "ML_Risky": [...], ... }``

Each label maps to a list of player names that satisfy its condition.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import MantraIbridoConfig

log = logging.getLogger(__name__)


def compute_hybrid_classifications(
    players_ibrido: list[dict[str, Any]],
    config: MantraIbridoConfig | None = None,
) -> dict[str, list[str]]:
    """Assign hybrid labels to each player and return name-grouped results.

    Parameters
    ----------
    players_ibrido:
        Player list enriched by :func:`~scoring.compute_hybrid_scores`.
    config:
        Active config (used for thresholds).  If ``None``, defaults are used.

    Returns
    -------
    dict mapping label name → sorted list of player names.
    """
    # Use a dummy config with defaults if none provided (avoids circular import).
    if config is None:
        from .config import MantraIbridoConfig as _C

        config = _C()

    labels: dict[str, list[str]] = {
        "ML_Confirmed": [],
        "ML_Risky": [],
        "ML_Boosted": [],
        "ML_Top": [],
        "Contradiction": [],
        "Minutes_Risk": [],
        "Best_Value": [],
        "Sleeper": [],
    }

    for p in players_ibrido:
        name = str(p.get("player_name", ""))

        # Skip players without ML data — hybrid labels don't apply.
        if not p.get("has_ml_data"):
            continue

        predicted = p.get("predicted_fantavoto")
        confidence = p.get("confidenceScore")
        expected_min = p.get("expected_minutes")
        ml_boost = p.get("mlBoost")
        fp_gap = p.get("fpGap")
        fp_corr = p.get("FP_Corr")
        ml_score_norm = p.get("ml_score_norm")
        vr = p.get("VR")
        fp_ibrido = p.get("fpIbrido")

        # Safe numeric conversions
        predicted = float(predicted) if predicted is not None else None
        confidence = float(confidence) if confidence is not None else None
        expected_min = float(expected_min) if expected_min is not None else None
        ml_boost = float(ml_boost) if ml_boost is not None else None
        fp_gap = float(fp_gap) if fp_gap is not None else None
        fp_corr = float(fp_corr) if fp_corr is not None else None
        ml_score_norm = float(ml_score_norm) if ml_score_norm is not None else None
        vr = float(vr) if vr is not None else None
        fp_ibrido = float(fp_ibrido) if fp_ibrido is not None else None

        # ── ML_Confirmed ─────────────────────────────────────────────────────
        # High predicted value + decent confidence + enough minutes
        if (
            predicted is not None
            and predicted > 6.5
            and confidence is not None
            and confidence >= config.CONFIDENZA_SOGLIA
            and expected_min is not None
            and expected_min > 1500
        ):
            labels["ML_Confirmed"].append(name)

        # ── ML_Risky ──────────────────────────────────────────────────────────
        # Very low confidence → unreliable prediction
        if confidence is not None and confidence < 50.0:
            labels["ML_Risky"].append(name)

        # ── ML_Boosted ────────────────────────────────────────────────────────
        # ML significantly above role mean AND player is not already top-rated
        # by MANTRA (FP_Corr not high) → "hidden gem" signal
        if (
            ml_boost is not None
            and ml_boost > config.ML_BOOST_SOGLIA
            and fp_corr is not None
            and fp_corr < config.ML_BOOST_FP_CORR_MAX
        ):
            labels["ML_Boosted"].append(name)

        # ── ML_Top ────────────────────────────────────────────────────────────
        # High predicted value + ML above role mean → top player
        if (
            predicted is not None
            and predicted >= config.ML_TOP_PRED_MIN
            and ml_boost is not None
            and ml_boost >= config.ML_TOP_BOOST_MIN
        ):
            labels["ML_Top"].append(name)

        # ── Contradiction ─────────────────────────────────────────────────────
        # Strong disagreement between MANTRA and ML
        if fp_gap is not None and abs(fp_gap) > config.SOGLIA_GAP_ALERT:
            labels["Contradiction"].append(name)

        # ── Minutes_Risk ──────────────────────────────────────────────────────
        if expected_min is not None and expected_min < config.MINUTES_RISK_MAX:
            labels["Minutes_Risk"].append(name)

        # ── Best_Value ────────────────────────────────────────────────────────
        # High VR (Value Rating) and decent hybrid score
        if (
            vr is not None
            and vr >= config.BEST_VALUE_VR_MIN
            and fp_ibrido is not None
            and fp_ibrido >= config.BEST_VALUE_FP_IBRIDO_MIN
        ):
            labels["Best_Value"].append(name)

        # ── Sleeper ───────────────────────────────────────────────────────────
        # Low MANTRA score but decent ML prediction → could be undervalued
        if (
            fp_corr is not None
            and fp_corr < config.SLEEPER_FP_CORR_MAX
            and ml_score_norm is not None
            and ml_score_norm > config.SLEEPER_ML_NORM_MIN
        ):
            labels["Sleeper"].append(name)

    log.info(
        "Hybrid classifications: %s",
        {k: len(v) for k, v in labels.items()},
    )
    return labels
