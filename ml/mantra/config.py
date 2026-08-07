"""Calibratable coefficients and thresholds for the MANTRA scoring engine.

All values are parametrisable via a single dataclass so the operator can
tune them without touching individual module files.  The defaults follow
the MANTRA v3.1 specification with the modifications agreed in the plan:

  - P3 weight raised to 0.30 for >= 10 % effective FP impact.
  - P1 / P2 lowered to 0.25 each.
  - Budget dynamic (no fixed blocks).
  - Low Cost threshold 15 credits (parametrisable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MantraConfig:
    """Global configuration for the MANTRA scoring pipeline."""

    # ── Pillar weights (must sum to 1.0) ─────────────────────────────────────
    PESO_P1: float = 0.25
    PESO_P2: float = 0.25
    PESO_P3: float = 0.30   # raised from 0.20 for >= 10 % effective impact
    PESO_P4: float = 0.20

    # ── Minutes thresholds ───────────────────────────────────────────────────
    SOGLIA_MINUTI_MIN: int = 450
    SOGLIA_MINUTI_MAX: int = 2700

    # ── Pool / standardisation ───────────────────────────────────────────────
    SOGLIA_POOL: int = 20       # min players in a role pool before merging
    CAP_K: float = 6.0          # cap for the tanh stretching factor

    # ── Hero factor ──────────────────────────────────────────────────────────
    FATTORE_EROE_MIN: float = 0.6
    FATTORE_EROE_MAX: float = 1.6

    # ── Role-specific coefficients for P3 (Peso Squadra) ─────────────────────
    COEFF_BASE: dict[str, float] = field(default_factory=lambda: {
        "Por": 0.0025,
        "Dc": 0.003, "B": 0.003, "Dd": 0.003, "Ds": 0.003,
        "E": 0.0035, "M": 0.0035,
        "C": 0.0038,
        "T": 0.0042, "W": 0.0042,
        "A": 0.004, "Pc": 0.004,
    })

    # ── Flexibility bonus ────────────────────────────────────────────────────
    FLESSIBILITA_1: float = 1.00
    FLESSIBILITA_2: float = 1.05
    FLESSIBILITA_3: float = 1.08

    # ── Classification thresholds ────────────────────────────────────────────
    LOW_COST_SOGLIA: float = 15.0          # Prezzo_Massimo under this = low cost
    GIOVANE_ETA_MAX: int = 23              # age <= this = young
    # Fixed thresholds — used as-is in FASE7_THRESHOLD_MODE="absolute", and as
    # the small-pool fallback in "percentile" mode (pools under SOGLIA_POOL).
    TOP_FP_SOGLIA: float = 80.0
    # Absolute VR floor for TOP (small-pool / absolute mode fallback).
    # Lower than AFFARE_VR_SOGLIA: TOP requires quality + non-cheap VR, not
    # necessarily "bargain" VR.
    TOP_VR_SOGLIA: float = 100.0
    AFFARE_FP_SOGLIA: float = 60.0
    AFFARE_VR_SOGLIA: float = 140.0
    SCOMMESSA_FP_SOGLIA: float = 50.0
    SCOMMESSA_VR_SOGLIA: float = 130.0
    CERTEZZA_STAGIONI: int = 2
    CERTEZZA_PR: float = 0.70
    CERTEZZA_P1: float = 55.0
    SOPRAVALUTATO_VR: float = 80.0
    GIUSTO_VR_MIN: float = 90.0
    GIUSTO_VR_MAX: float = 110.0

    # ── Fase 7 threshold mode ─────────────────────────────────────────────────
    # "percentile": TOP/AFFARE/SCOMMESSA/SOPRAVALUTATO/GIUSTO thresholds are
    #   computed per role-pool (see ml/mantra/fase7.py) so e.g. TOP means "top
    #   ~10% of FP_Mantra within your role", not one global number that may
    #   favor/disadvantage specific roles. "absolute": always use the fixed
    #   thresholds above (pre-percentile behavior, useful as a rollback knob).
    FASE7_THRESHOLD_MODE: Literal["percentile", "absolute"] = "percentile"
    # Raised from 0.85 → 0.90 so TOP is the top ~10% of the role pool, not ~15%.
    TOP_FP_PERCENTILE: float = 0.90
    # TOP also requires VR at least around the role-pool median (anti-lowcost-only).
    TOP_VR_PERCENTILE: float = 0.55
    AFFARE_FP_PERCENTILE: float = 0.65
    AFFARE_VR_PERCENTILE: float = 0.85
    SCOMMESSA_FP_PERCENTILE: float = 0.35
    SCOMMESSA_VR_PERCENTILE: float = 0.80
    SOPRAVALUTATO_VR_PERCENTILE: float = 0.35
    GIUSTO_VR_PERCENTILE_MIN: float = 0.42
    GIUSTO_VR_PERCENTILE_MAX: float = 0.58
    CERTEZZA_DV_PERCENTILE: float = 0.50   # 0.50 = median (unchanged historical behavior)

    # ── TOP external gates (optional columns on the player DataFrame) ─────────
    # If the column is missing or the cell is null, the gate is skipped
    # (does not block TOP). If present, the value must pass the threshold.
    # Expert rating scale follows the scraped source (e.g. 1-10 style where
    # Muric=4 is weak); raise/lower TOP_EXPERT_MIN if your source uses 1-5 stars.
    TOP_EXPERT_MIN: float = 6.0
    # Minimum predicted_next_fantavoto (or predicted_fantavoto fallback) by
    # primary MANTRA role. Used only when the column is present and non-null.
    NEXT_FANTAVOTO_MIN_BY_ROLE: dict[str, float] = field(default_factory=lambda: {
        "Por": 5.7,
        "Dc": 5.9, "Dd": 5.9, "Ds": 5.9, "B": 5.9,
        "C": 6.1, "M": 6.1, "E": 6.1,
        "W": 6.3, "T": 6.3,
        "A": 6.5, "Pc": 6.5,
    })

    # ── Budget ───────────────────────────────────────────────────────────────
    BUDGET_TOTALE: int = 500

    # ── PS_corretto weights ──────────────────────────────────────────────────
    PS_TEAM_RANK_WEIGHT: float = 0.27
    PS_PREV_POINTS_WEIGHT: float = 0.22
    PS_GOAL_DIFF_WEIGHT: float = 0.17
    PS_AVG_RATING_WEIGHT: float = 0.17
    PS_SQUAD_VALUE_WEIGHT: float = 0.17

    def __post_init__(self) -> None:
        total = self.PESO_P1 + self.PESO_P2 + self.PESO_P3 + self.PESO_P4
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Pillar weights must sum to 1.0, got {total}"
            )
        ps_total = (
            self.PS_TEAM_RANK_WEIGHT
            + self.PS_PREV_POINTS_WEIGHT
            + self.PS_GOAL_DIFF_WEIGHT
            + self.PS_AVG_RATING_WEIGHT
            + self.PS_SQUAD_VALUE_WEIGHT
        )
        if abs(ps_total - 1.0) > 1e-6:
            raise ValueError(
                f"PS_corretto weights must sum to 1.0, got {ps_total}"
            )
