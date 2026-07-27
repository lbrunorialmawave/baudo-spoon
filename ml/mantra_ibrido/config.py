"""Config dataclass for the hybrid MANTRA+ML scorer.

All weights and thresholds are user-configurable via the admin panel.
Defaults assume a 50/50 blend for initial testing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MantraIbridoConfig:
    # ── FP_Ibrido weights (must sum to 1.0) ──────────────────────────────────
    PESO_MANTRA: float = 0.5
    PESO_ML: float = 0.5

    # ── Confidence_Score weights (must sum to 1.0) ───────────────────────────
    # NOTE: reliability_weight was removed because it is not serialised in the
    #       ML pipeline output (results_latest.json). It only exists in internal
    #       domain models (PlayerV2, Player).
    W_PREDICTION_STD: float = 0.6
    W_MINUTES: float = 0.4

    # ── Expected_Value ───────────────────────────────────────────────────────
    # Multiplier applied after FP_Ibrido_voto * partite_attese.
    # Default 1.0 = no correction. Lower to penalise optimistic estimates.
    EV_SCALE_FACTOR: float = 1.0

    # ── Classification thresholds ─────────────────────────────────────────────
    SOGLIA_CONFIDENZA_MIN: float = 0.3
    SOGLIA_GAP_ALERT: float = 25.0
