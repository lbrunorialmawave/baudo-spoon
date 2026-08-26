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
    # Only TOP and CERTEZZA still use a threshold: SCOMMESSA (Rendimento axis)
    # and AFFARE/GIUSTO/SOPRAVALUTATO (Prezzo/Valore axis) are gap-based (see
    # SCOMMESSA_GAP_MIN / GIUSTO_GAP_BAND below) and need no per-role fallback.
    TOP_FP_SOGLIA: float = 80.0
    # Absolute VR floor for TOP (small-pool / absolute mode fallback).
    # Lower than the old AFFARE_VR_SOGLIA: TOP requires quality + non-cheap
    # VR, not necessarily "bargain" VR.
    TOP_VR_SOGLIA: float = 95.0
    CERTEZZA_STAGIONI: int = 2
    CERTEZZA_PR: float = 0.70
    CERTEZZA_P1: float = 55.0
    # Forward-looking CERTEZZA leg: EWMA of player_matchday_status.probability
    # (0-100). Lets a newly-arrived player with a locked-in starting spot
    # qualify even with Stagioni_IT below CERTEZZA_STAGIONI.
    CERTEZZA_TITOLARITA_SOGLIA: float = 75.0
    TITOLARITA_EWMA_LAMBDA: float = 0.65
    TITOLARITA_EWMA_WINDOW: int = 5
    # Second forward-looking CERTEZZA leg, from Gruppo Esperti (titolarita +
    # salute, 1-10 scale each, no DV gate — see ml/mantra/fase7.py). A
    # dedicated (lower) threshold than CERTEZZA_TITOLARITA_SOGLIA: the
    # experts' scale only has 10 discrete values, coarser than the EWMA one.
    CERTEZZA_ESPERTI_SOGLIA: float = 70.0
    CERTEZZA_ESPERTI_TITOLARITA_PESO: float = 0.7   # salute weight = 1 - this

    # ── Fase 7 gap-based axes ─────────────────────────────────────────────────
    # Rendimento/Affidabilità axis (TOP > CERTEZZA > SCOMMESSA): SCOMMESSA is
    # the divergence between a player's VR percentile and his raw-FP
    # percentile within his role pool, not two independent absolute
    # thresholds (which almost never overlap in practice). It also requires
    # a cheap quotation (SCOMMESSA_PREZZO_PERCENTILE_MAX): a "scommessa" is a
    # cheap gamble with hidden upside by definition — without a price gate,
    # the label can land on an already-pricey player, which defeats its
    # purpose (validated against real 2026 auction data: the two ungated
    # candidates were priced above their role's median).
    SCOMMESSA_GAP_MIN: float = 15.0
    SCOMMESSA_PREZZO_PERCENTILE_MAX: float = 40.0
    # Gruppo Esperti quality boost (bonus + media_voto, 1-10 scale each) on
    # the SCOMMESSA gap: a marginal statistical gap can still qualify when
    # the expert panel rates the player's quality well above average. Never
    # negative (floor 0) and never blocking — missing data just means no
    # boost, same "null = gate skipped" convention as the rest of this
    # module. NEUTRAL is the 0-100 quality-index value below which no boost
    # applies (~6/10 raw); BOOST_MAX (2/3 of SCOMMESSA_GAP_MIN) can tip a
    # near-miss over the line, not manufacture a gap from nothing.
    SCOMMESSA_QUALITY_NEUTRAL: float = 55.0
    SCOMMESSA_QUALITY_BOOST_MAX: float = 10.0
    SCOMMESSA_QUALITY_BONUS_PESO: float = 0.5   # media_voto weight = 1 - this
    # Prezzo/Valore axis (AFFARE / GIUSTO / SOPRAVALUTATO): three contiguous
    # bands of the gap between a player's quotation percentile and his
    # FP_Mantra percentile within his role pool. |gap| <= this band = GIUSTO.
    GIUSTO_GAP_BAND: float = 15.0
    # Gruppo Esperti TOTALE (/50 scale) blended into the Prezzo/Valore
    # "quality" anchor alongside FP_Mantra: a player with a temporarily poor
    # backward-looking VR/FP_Mantra (bad luck — missed penalties, hit the
    # woodwork) but a strong expert TOTALE shouldn't be flagged SOPRAVALUTATO
    # on the statistical read alone. Equal weight with FP_Mantra: validated
    # against real 2026 auction data — a conservative 0.25 weight was not
    # enough to rescue an expert-loved, most-expensive-in-role player from a
    # false SOPRAVALUTATO (gap stayed at 22.5 vs. the 15.0 band); 0.5 does
    # (gap 15.6, at the edge). 0 = ignore experts entirely (pre-Gruppo-Esperti
    # behavior); falls back to FP_Mantra-only automatically when TOTALE is
    # missing for a player.
    PREZZO_EXPERT_TOTALE_WEIGHT: float = 0.5

    # ── Fase 7 threshold mode ─────────────────────────────────────────────────
    # "percentile": TOP thresholds (and CERTEZZA's DV leg) are computed per
    #   role-pool (see ml/mantra/fase7.py) so e.g. TOP means "top ~10% of
    #   FP_Mantra within your role", not one global number that may
    #   favor/disadvantage specific roles. "absolute": always use the fixed
    #   thresholds above (pre-percentile behavior, useful as a rollback knob).
    FASE7_THRESHOLD_MODE: Literal["percentile", "absolute"] = "percentile"
    # Raised from 0.85 → 0.90 so TOP is the top ~10% of the role pool, not ~15%.
    TOP_FP_PERCENTILE: float = 0.88
    # TOP also requires VR at least around the role-pool median (anti-lowcost-only).
    TOP_VR_PERCENTILE: float = 0.40
    CERTEZZA_DV_PERCENTILE: float = 0.50   # 0.50 = median (unchanged historical behavior)

    # ── TOP external gates (optional columns on the player DataFrame) ─────────
    # If the column is missing or the cell is null, the gate is skipped
    # (does not block TOP). If present, the value must pass the threshold.
    # Expert rating scale follows the scraped source (e.g. 1-10 style where
    # Muric=4 is weak); raise/lower TOP_EXPERT_MIN if your source uses 1-5 stars.
    TOP_EXPERT_MIN: float = 3.0
    # Minimum predicted_next_fantavoto (or predicted_fantavoto fallback) by
    # primary MANTRA role. Used only when the column is present and non-null.
    NEXT_FANTAVOTO_MIN_BY_ROLE: dict[str, float] = field(default_factory=lambda: {
        "Por": 5.5,
        "Dc": 5.7, "Dd": 5.7, "Ds": 5.7, "B": 5.7,
        "C": 5.9, "M": 5.9, "E": 5.9,
        "W": 6.0, "T": 6.0,
        "A": 6.2, "Pc": 6.2,
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
