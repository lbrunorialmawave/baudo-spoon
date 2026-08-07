"""Config dataclass for the hybrid MANTRA+ML scorer.

All weights and thresholds are user-configurable via the admin panel.
Thresholds calibrated 2026-07-28 based on real data percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MantraIbridoConfig:
    # ── FP_Ibrido weights (must sum to 1.0) ──────────────────────────────────
    PESO_MANTRA: float = 0.5
    PESO_ML: float = 0.5

    # ── Confidence_Score weights (must sum to 1.0) ───────────────────────────
    W_PREDICTION_STD: float = 0.6
    W_MINUTES: float = 0.4

    # ── Expected_Value ───────────────────────────────────────────────────────
    EV_SCALE_FACTOR: float = 1.0

    # ── Classification thresholds (calibrated 2026-07-28) ────────────────────
    #
    # Percentile reference (n=351 players with ML data):
    #   confidence: P25=57.5  P50=58.3  P75=58.9
    #   mlBoost:    P25=39.3  P50=46.9  P75=59.6  P90=68.7
    #   fpGap:      P25=-6.9  P50=7.8   P75=25.6  P90=36.1
    #   FP_Corr:    P25=29.0  P50=45.3  P75=63.3
    #   ml_norm:    P25=41.7  P50=46.5  P75=53.1
    #   VR:         P25=72.2  P50=93.7  P75=111.6 P90=126.6
    #   predicted:  P25=6.1   P50=6.3   P75=6.7   P90=7.0

    # Minimum confidence for ML_Confirmed (confidence P25 ≈ 57.5)
    CONFIDENZA_SOGLIA: float = 57.0

    # ML_Boosted: ML prediction well above role mean (mlBoost P90 ≈ 68.7)
    # AND FP_Corr below P50 (player not already known as top)
    ML_BOOST_SOGLIA: float = 70.0
    ML_BOOST_FP_CORR_MAX: float = 60.0

    # ML_Top: high predicted value AND strong mlBoost (top players)
    ML_TOP_PRED_MIN: float = 6.7  # predicted P75
    ML_TOP_BOOST_MIN: float = 65.0  # mlBoost ~P85

    # Contradiction: strong MANTRA vs ML disagreement (|gap| > P85 ≈ 33)
    SOGLIA_GAP_ALERT: float = 30.0

    # Sleeper: low MANTRA score (FP_Corr < P25) but decent ML (ml_norm > P50)
    SLEEPER_FP_CORR_MAX: float = 30.0
    SLEEPER_ML_NORM_MIN: float = 45.0

    # Best_Value: high VR (VR > P75 ≈ 112) and decent hybrid score
    BEST_VALUE_VR_MIN: float = 110.0
    BEST_VALUE_FP_IBRIDO_MIN: float = 50.0

    # Minutes_Risk: low expected minutes
    MINUTES_RISK_MAX: float = 900.0
