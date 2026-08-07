"""Hybrid scoring logic — combines MANTRA pillars with ML predictions.

All scores are computed server-side so the frontend receives ready-to-display
values without additional calculation.

Score reference
---------------
*   ``ML_score_norm`` — predicted_fantavoto (scale 4-9) mapped to 0-100
*   ``FP_Ibrido`` — weighted average of ``FP_Corr`` and ``ML_score_norm``
*   ``Confidence_Score`` 0-100 — how reliable the estimate is
*   ``ML_Boost`` 0-100 — z-score of predicted within role pool, centred at 50
*   ``FP_Gap`` — ``FP_Corr - ML_score_norm`` (both 0-100, directly comparable)
*   ``Expected_Value`` — ``FP_Ibrido_voto * partite_attese * EV_SCALE_FACTOR``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import MantraIbridoConfig

log = logging.getLogger(__name__)

# ── Normalisation constants ───────────────────────────────────────────────────
# predicted_fantavoto typical range: 4.0 (very poor) – 9.0 (exceptional)
_FANTAVOTO_MIN = 4.0
_FANTAVOTO_MAX = 9.0
_FANTAVOTO_RANGE = _FANTAVOTO_MAX - _FANTAVOTO_MIN  # = 5.0

# Expected minutes ceiling for normalisation (full season ≈ 2700').
_MAX_EXPECTED_MINUTES = 2700.0


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_hybrid_scores(
    players_arricchiti: list[dict[str, Any]],
    config: MantraIbridoConfig,
) -> list[dict[str, Any]]:
    """Add hybrid score fields to each player record (mutates in place).

    Parameters
    ----------
    players_arricchiti:
        List of player dicts from :func:`~merger.merge_datasets`.
    config:
        Active hybrid configuration (weights, thresholds).

    Returns
    -------
    The same list with extra keys set on each dict.
    """
    # ── Pre-compute role-pool statistics for ML_Boost ─────────────────────────
    # Collect all predicted values per role (only those with ML data).
    role_predicted: dict[str, list[float]] = {}
    for p in players_arricchiti:
        pred = p.get("predicted_fantavoto")
        ruolo = p.get("ruolo_primario") or p.get("canonicalRole") or "unknown"
        if pred is not None and p.get("has_ml_data"):
            role_predicted.setdefault(ruolo, []).append(float(pred))

    role_mean: dict[str, float] = {}
    role_std: dict[str, float] = {}
    for ruolo, vals in role_predicted.items():
        arr = np.array(vals)
        role_mean[ruolo] = float(np.mean(arr))
        role_std[ruolo] = float(np.std(arr)) if len(arr) > 1 else 1.0

    # ── Score each player ─────────────────────────────────────────────────────
    for p in players_arricchiti:
        if not p.get("has_ml_data"):
            # No ML data — use MANTRA-only fallback
            fp_corr = p.get("FP_Corr")
            p["fpIbrido"] = fp_corr if fp_corr is not None else None
            p["ml_score_norm"] = None
            p["confidenceScore"] = 0.0
            p["mlBoost"] = None
            p["fpGap"] = None
            p["expectedValue"] = None
            p["hybridLabels"] = []
            continue

        p["hybridLabels"] = []

        predicted = p["predicted_fantavoto"]
        if predicted is None:
            # Defensive: should not happen when has_ml_data is True, but guard.
            p["fpIbrido"] = p.get("FP_Corr")
            p["ml_score_norm"] = None
            p["confidenceScore"] = 0.0
            p["mlBoost"] = None
            p["fpGap"] = None
            p["expectedValue"] = None
            p["hybridLabels"] = []
            continue

        predicted = float(predicted)
        fp_corr = p.get("FP_Corr")
        fp_corr = float(fp_corr) if fp_corr is not None else 50.0
        pred_std = p.get("prediction_std")
        pred_std = float(pred_std) if pred_std is not None else 0.5
        expected_min = p.get("expected_minutes")
        expected_min = float(expected_min) if expected_min is not None else 0.0

        ruolo = p.get("ruolo_primario") or p.get("canonicalRole") or "unknown"

        # ── ML_score_norm (predicted 4-9 → 0-100) ───────────────────────────
        ml_norm = _clip(
            (predicted - _FANTAVOTO_MIN) / _FANTAVOTO_RANGE * 100.0,
            0.0,
            100.0,
        )

        # ── FP_Ibrido ────────────────────────────────────────────────────────
        fp_ibrido = fp_corr * config.PESO_MANTRA + ml_norm * config.PESO_ML

        # ── Confidence_Score 0-100 ───────────────────────────────────────────
        std_term = 1.0 / (1.0 + pred_std)
        min_term = min(expected_min / _MAX_EXPECTED_MINUTES, 1.0)
        confidence = (
            std_term * config.W_PREDICTION_STD + min_term * config.W_MINUTES
        ) * 100.0

        # ── ML_Boost (z-score centred at 50, σ=15) ──────────────────────────
        mu = role_mean.get(ruolo, 5.5)
        sigma = role_std.get(ruolo, 1.0)
        z = (predicted - mu) / max(sigma, 0.01)
        ml_boost = _clip(50.0 + z * 15.0, 0.0, 100.0)

        # ── FP_Gap (both 0-100) ──────────────────────────────────────────────
        fp_gap = fp_corr - ml_norm

        # ── Expected_Value (Punti Stagione Attesi) ───────────────────────────
        # FP_Ibrido_voto riporta lo score 0-100 in scala voto reale (4-10).
        fp_ibrido_voto = 4.0 + (fp_ibrido / 100.0) * 6.0
        partite_attese = expected_min / 90.0
        expected_value = fp_ibrido_voto * partite_attese * config.EV_SCALE_FACTOR

        # ── Write back ───────────────────────────────────────────────────────
        p["ml_score_norm"] = round(ml_norm, 2)
        p["fpIbrido"] = round(fp_ibrido, 2)
        p["confidenceScore"] = round(confidence, 2)
        p["mlBoost"] = round(ml_boost, 2)
        p["fpGap"] = round(fp_gap, 2)
        p["expectedValue"] = round(expected_value, 2)

    return players_arricchiti
