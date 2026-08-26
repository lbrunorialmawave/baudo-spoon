"""Tests for Fase 7 classification (ml/mantra/fase7.py::classify_fase7).

Fase7 has two independent axes:
- Rendimento/Affidabilità: TOP > CERTEZZA > SCOMMESSA (first match wins)
- Prezzo/Valore: AFFARE / GIUSTO / SOPRAVALUTATO (three contiguous bands of
  one gap — no cascade ordering needed, they can't overlap)

A player can carry a label on both axes independently, one, or neither.
Covers each label in isolation, the rule-priority order within the
Rendimento axis, the percentile-per-role-pool threshold mode (TOP/CERTEZZA
only — SCOMMESSA and the Prezzo axis are pure percentile gaps), and the
Fase7_*_Motivo "why no classification" explanations.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml.mantra.config import MantraConfig
from ml.mantra.fase7 import classify_fase7


@pytest.fixture
def cfg():
    return MantraConfig()


def _base_df(n: int, ruolo: str = "C") -> pd.DataFrame:
    """Minimal DataFrame with the columns classify_fase7 reads directly.

    A single role, no Pz1 column (never quoted → Prezzo axis stays
    unclassified) — small enough that TOP/CERTEZZA percentile mode falls
    back to the fixed absolute thresholds, so these fixtures behave
    identically regardless of cfg.FASE7_THRESHOLD_MODE.
    """
    return pd.DataFrame({
        "ruolo_primario": [ruolo] * n,
        "Stagioni_IT": [0] * n,
        "Pr": [0.0] * n,
        "DV": [5.0] * n,
    })


def _classify(df, fp, fp_mantra, vr, p1, cfg):
    """Run classify_fase7, coercing plain lists to Series aligned to df."""
    return classify_fase7(
        df,
        pd.Series(fp, index=df.index),
        pd.Series(fp_mantra, index=df.index),
        pd.Series(vr, index=df.index),
        pd.Series(p1, index=df.index),
        cfg,
    )


def _rend(df, fp, fp_mantra, vr, p1, cfg):
    """Convenience wrapper for tests that only care about the Rendimento
    axis label."""
    label_rend, _, _, _, _, _ = _classify(df, fp, fp_mantra, vr, p1, cfg)
    return label_rend


class TestRendimentoIndividualLabels:
    def test_top(self, cfg):
        df = _base_df(1)
        # VR must clear TOP_VR_SOGLIA (95) in absolute/small-pool mode
        result = _rend(df, fp=[70], fp_mantra=[85], vr=[101], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_certezza_historical(self, cfg):
        """CERTEZZA (historical leg) requires Stagioni_IT>=2, Pr>=0.70,
        DV<=role-pool median, P1>=55."""
        df = pd.DataFrame({
            "ruolo_primario": ["C", "C", "C"],
            "Stagioni_IT": [2, 2, 2],
            "Pr": [0.80, 0.75, 0.85],
            "DV": [3.0, 3.0, 3.0],
        })
        result = _rend(
            df, fp=[55, 55, 55], fp_mantra=[55, 55, 55], vr=[95, 95, 95],
            p1=[60, 58, 65], cfg=cfg,
        )
        assert (result == "CERTEZZA").all()

    def test_certezza_fails_below_stagioni_threshold(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [1], "Pr": [0.80], "DV": [3.0],
        })
        result = _rend(df, fp=[55], fp_mantra=[55], vr=[95], p1=[60], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_certezza_fails_below_pr_threshold(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [2], "Pr": [0.50], "DV": [3.0],
        })
        result = _rend(df, fp=[55], fp_mantra=[55], vr=[95], p1=[60], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_certezza_forward_via_titolarita_attesa(self, cfg):
        """A newly-arrived player with no Serie A history but a locked-in
        starting spot (high EWMA titolarita_attesa) still qualifies for
        CERTEZZA via the forward-looking OR leg."""
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [0], "Pr": [0.0], "DV": [3.0],
            "titolarita_attesa": [90.0],
        })
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[95], p1=[30], cfg=cfg)
        assert result.iloc[0] == "CERTEZZA"

    def test_certezza_forward_fails_below_titolarita_soglia(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [0], "Pr": [0.0], "DV": [3.0],
            "titolarita_attesa": [50.0],  # below CERTEZZA_TITOLARITA_SOGLIA (75)
        })
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[95], p1=[30], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_scommessa_gap(self, cfg):
        """SCOMMESSA is a gap between VR percentile and raw-FP percentile
        within the role pool, not two independent absolute thresholds — AND
        the candidate must actually be cheap (a "scommessa" is a cheap
        gamble by definition)."""
        n = 6
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
            "Pz1": [50, 60, 70, 80, 90, 5],
        })
        # Candidate (last row): lowest raw FP, highest VR, cheapest price.
        fp = [80, 78, 76, 74, 72, 20]
        fp_mantra = [40] * n  # keep everyone well below TOP_FP_SOGLIA (80)
        vr = [90, 92, 94, 96, 98, 130]
        p1 = [50] * n
        result = _rend(df, fp=fp, fp_mantra=fp_mantra, vr=vr, p1=p1, cfg=cfg)
        assert result.iloc[-1] == "SCOMMESSA"

    def test_scommessa_requires_cheap_price(self, cfg):
        """Same gap as an otherwise-qualifying SCOMMESSA candidate, but
        priced at the top of the role pool: must NOT be labeled SCOMMESSA —
        a "scommessa" that costs more than everyone else in its role isn't
        a gamble, it's a bad buy. Regression for real 2026 auction data
        (Cutrone/Frattesi) where an ungated gap rule produced SCOMMESSA tags
        on above-median-priced players."""
        n = 6
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
            "Pz1": [5, 10, 15, 20, 25, 100],  # candidate is the most expensive
        })
        fp = [80, 78, 76, 74, 72, 20]
        fp_mantra = [40] * n
        vr = [90, 92, 94, 96, 98, 130]
        p1 = [50] * n
        result = _rend(df, fp=fp, fp_mantra=fp_mantra, vr=vr, p1=p1, cfg=cfg)
        assert result.iloc[-1] != "SCOMMESSA"

    def test_none_when_no_rendimento_rule_matches(self, cfg):
        n = 3
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
        })
        # FP and VR move in lockstep -> gap is always ~0, never > SCOMMESSA_GAP_MIN
        result = _rend(
            df, fp=[40, 45, 50], fp_mantra=[35, 35, 35], vr=[80, 82, 84],
            p1=[50, 50, 50], cfg=cfg,
        )
        assert result.isna().all()


class TestPrezzoAxis:
    def test_affare_and_sopravalutato_opposite_ends(self, cfg):
        """Gap uses FP_Mantra (quality), not VR — see module docstring:
        comparing price against VR would double-count cost, since VR
        already discounts expensive/strong players via Fattore_Eroe."""
        df = pd.DataFrame({
            "ruolo_primario": ["C", "C"],
            "Stagioni_IT": [0, 0], "Pr": [0.0, 0.0], "DV": [5.0, 5.0],
            "Pz1": [10, 100],
        })
        _, _, _, label_prezzo, _, gap_prezzo = _classify(
            df, fp=[40, 40], fp_mantra=[130, 40], vr=[90, 90], p1=[50, 50], cfg=cfg,
        )
        assert label_prezzo.iloc[0] == "AFFARE"  # cheap, high quality
        assert label_prezzo.iloc[1] == "SOPRAVALUTATO"  # expensive, low quality
        assert gap_prezzo.iloc[0] < 0
        assert gap_prezzo.iloc[1] > 0

    def test_giusto_when_price_tracks_quality(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * 3,
            "Stagioni_IT": [0] * 3, "Pr": [0.0] * 3, "DV": [5.0] * 3,
            "Pz1": [10, 50, 90],
        })
        # Price and FP_Mantra rank in the same order -> gap ~ 0 for all three.
        _, _, _, label_prezzo, _, _ = _classify(
            df, fp=[40] * 3, fp_mantra=[60, 100, 140], vr=[90] * 3, p1=[50] * 3, cfg=cfg,
        )
        assert (label_prezzo == "GIUSTO").all()

    def test_expensive_elite_player_is_not_sopravalutato(self, cfg):
        """Regression: an elite AND expensive player (highest price, highest
        FP_Mantra in the pool) must land GIUSTO on the Prezzo axis, not
        SOPRAVALUTATO — even though VR itself may be comparatively lower for
        him (Fattore_Eroe dampens VR for high-CP/expensive players by
        design; that's a Rendimento-axis nuance, not a pricing error)."""
        n = 5
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
            "Pz1": [10, 20, 30, 40, 100],
        })
        fp = [40] * n
        fp_mantra = [40, 50, 60, 70, 95]  # candidate: also highest quality
        vr = [90, 92, 94, 96, 60]  # candidate: dampened VR despite being the best
        p1 = [50] * n
        _, _, _, label_prezzo, _, _ = _classify(
            df, fp=fp, fp_mantra=fp_mantra, vr=vr, p1=p1, cfg=cfg,
        )
        assert label_prezzo.iloc[-1] == "GIUSTO"

    def test_none_when_never_quoted(self, cfg):
        df = _base_df(1)  # no Pz1 column at all
        _, _, _, label_prezzo, motivo_prezzo, gap_prezzo = _classify(
            df, fp=[40], fp_mantra=[40], vr=[95], p1=[50], cfg=cfg,
        )
        assert pd.isna(label_prezzo.iloc[0])
        assert pd.isna(gap_prezzo.iloc[0])
        assert "quotazione" in motivo_prezzo.iloc[0].lower()


class TestAxesAreIndependent:
    def test_certezza_and_affare_can_coexist(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C", "C"],
            "Stagioni_IT": [2, 2], "Pr": [0.9, 0.9], "DV": [1.0, 1.0],
            "Pz1": [10, 100],
        })
        label_rend, _, _, label_prezzo, _, _ = _classify(
            df, fp=[55, 55], fp_mantra=[55, 55], vr=[95, 95], p1=[60, 60], cfg=cfg,
        )
        assert (label_rend == "CERTEZZA").all()
        assert label_prezzo.iloc[0] == "AFFARE"
        assert label_prezzo.iloc[1] == "SOPRAVALUTATO"


class TestRendimentoRulePriority:
    def test_top_wins_over_certezza(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [2], "Pr": [0.90], "DV": [1.0],
        })
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[101], p1=[90], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_certezza_wins_over_scommessa(self, cfg):
        """CERTEZZA is evaluated before SCOMMESSA within the Rendimento axis."""
        n = 3
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [2, 0, 0],
            "Pr": [0.9, 0.0, 0.0],
            "DV": [1.0, 5.0, 5.0],
        })
        fp = [20, 60, 80]         # candidate (idx0) has the lowest raw FP
        fp_mantra = [40] * n
        vr = [140, 100, 60]       # candidate (idx0) has the highest VR
        p1 = [60, 50, 50]
        result = _rend(df, fp=fp, fp_mantra=fp_mantra, vr=vr, p1=p1, cfg=cfg)
        # idx0 would also satisfy the SCOMMESSA gap, but CERTEZZA wins.
        assert result.iloc[0] == "CERTEZZA"


class TestPercentileMode:
    def test_default_mode_is_percentile(self, cfg):
        assert cfg.FASE7_THRESHOLD_MODE == "percentile"

    def test_top_is_role_relative_not_global(self, cfg):
        """A role pool whose FP_Mantra never reaches the fixed absolute TOP
        threshold (80) still produces a TOP label in percentile mode,
        because the pool is large enough (>= SOGLIA_POOL) for TOP to be
        evaluated relative to the role's own distribution."""
        n = 25  # >= SOGLIA_POOL (20) -> percentile mode applies
        fp_mantra_values = list(np.linspace(30.0, 54.0, n))  # all well below 80
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
        })
        vr_values = list(np.linspace(80.0, 130.0, n))
        vr = pd.Series(vr_values)
        p1 = pd.Series([50.0] * n)
        fp_mantra = pd.Series(fp_mantra_values)

        label, *_ = classify_fase7(df, fp_mantra, fp_mantra, vr, p1, cfg)
        assert label.iloc[-1] == "TOP"  # highest FP_Mantra + high VR in the pool

        abs_cfg = replace(cfg, FASE7_THRESHOLD_MODE="absolute")
        label_abs, *_ = classify_fase7(df, fp_mantra, fp_mantra, vr, p1, abs_cfg)
        assert (label_abs == "TOP").sum() == 0  # nobody ever reaches the fixed 80 threshold

    def test_small_pool_falls_back_to_absolute(self, cfg):
        df = _base_df(1)
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[101], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"


class TestFase7Motivo:
    def test_rendimento_motivo_none_for_a_classified_player(self, cfg):
        df = _base_df(1)
        label, motivo, *_ = _classify(
            df, fp=[70], fp_mantra=[85], vr=[101], p1=[50], cfg=cfg,
        )
        assert label.iloc[0] == "TOP"
        assert motivo.iloc[0] is None

    def test_motivo_mentions_gap_when_no_rendimento_rule_matches(self, cfg):
        n = 3
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
        })
        _, motivo, *_ = _classify(
            df, fp=[40, 45, 50], fp_mantra=[35, 35, 35], vr=[80, 82, 84],
            p1=[50, 50, 50], cfg=cfg,
        )
        assert all("Gap Rendimento" in m for m in motivo)

    def test_quasi_certezza_names_the_single_failing_historical_condition(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [2], "Pr": [0.62], "DV": [3.0],
        })
        _, motivo, *_ = _classify(
            df, fp=[40], fp_mantra=[40], vr=[85], p1=[60], cfg=cfg,
        )
        assert "Quasi CERTEZZA" in motivo.iloc[0]
        assert "Pr" in motivo.iloc[0]

    def test_quasi_certezza_via_titolarita_attesa(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [0], "Pr": [0.0], "DV": [3.0],
            "titolarita_attesa": [70.0],  # soglia is 75, within 10 points
        })
        _, motivo, *_ = _classify(
            df, fp=[40], fp_mantra=[40], vr=[85], p1=[50], cfg=cfg,
        )
        assert "titolarità attesa" in motivo.iloc[0]

    def test_prezzo_motivo_mentions_missing_quotation(self, cfg):
        df = _base_df(1)
        *_, motivo_prezzo, _ = _classify(
            df, fp=[40], fp_mantra=[40], vr=[95], p1=[50], cfg=cfg,
        )
        assert "quotazione" in motivo_prezzo.iloc[0].lower()


class TestTopExternalGates:
    """TOP blocked by present ML / expert signals; nulls never block."""

    def test_expert_rating_blocks_top(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [4.0]  # below TOP_EXPERT_MIN (3.0 is passed, use a low one)
        df["expert_rating"] = [2.0]
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_expert_rating_passes_top(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [7.0]
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_missing_expert_does_not_block(self, cfg):
        df = _base_df(1)
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_ml_below_soglia_blocks_top(self, cfg):
        df = _base_df(1)
        df["ruolo_primario"] = ["Por"]
        df["predicted_next_fantavoto"] = [5.0]  # Por soglia 5.7
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_ml_above_soglia_allows_top(self, cfg):
        df = _base_df(1)
        df["ruolo_primario"] = ["Por"]
        df["predicted_next_fantavoto"] = [6.0]
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_vr_below_floor_blocks_top(self, cfg):
        df = _base_df(1)
        result = _rend(df, fp=[85], fp_mantra=[85], vr=[90], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_motivo_mentions_expert_block(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [2.0]
        label, motivo, *_ = _classify(
            df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg,
        )
        assert label.iloc[0] != "TOP"
        assert motivo.iloc[0] is not None
        assert "esperti" in motivo.iloc[0].lower() or "rating" in motivo.iloc[0].lower()


class TestCertezzaEspertiLeg:
    """Gruppo Esperti titolarita+salute as a second, independent
    forward-looking CERTEZZA leg (alongside the EWMA one)."""

    def test_idx_above_soglia_grants_certezza(self, cfg):
        df = _base_df(1)  # Stagioni_IT=0 -> certezza_storica always false here
        df["expert_titolarita"] = [9]
        df["expert_salute"] = [9]
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[90], p1=[30], cfg=cfg)
        assert result.iloc[0] == "CERTEZZA"

    def test_idx_below_soglia_does_not_grant_certezza(self, cfg):
        df = _base_df(1)
        df["expert_titolarita"] = [5]
        df["expert_salute"] = [5]
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[90], p1=[30], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_requires_both_subscores(self, cfg):
        """A single sub-score present (the other missing) must not grant
        CERTEZZA — the index needs both titolarita and salute."""
        df = _base_df(1)
        df["expert_titolarita"] = [10]  # expert_salute column absent entirely
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[90], p1=[30], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_ignores_dv_gate(self, cfg):
        """Unlike the historical/EWMA legs, the experts leg has no DV gate —
        it exists specifically for players who lack the statistical history
        DV is built from."""
        df = pd.DataFrame({
            "ruolo_primario": ["C"], "Stagioni_IT": [0], "Pr": [0.0],
            "DV": [99.0],  # deliberately terrible/default value
            "expert_titolarita": [10], "expert_salute": [10],
        })
        result = _rend(df, fp=[40], fp_mantra=[40], vr=[90], p1=[30], cfg=cfg)
        assert result.iloc[0] == "CERTEZZA"


def _boost_pool_df(pz1_candidate_cheap: float = 5) -> pd.DataFrame:
    """11-player 'C' pool (10-point percentile granularity) where the last
    row is the SCOMMESSA candidate: lowest raw FP, 2nd-lowest VR (gap=10,
    just under SCOMMESSA_GAP_MIN=15 without a boost), cheapest price."""
    n = 11
    return pd.DataFrame({
        "ruolo_primario": ["C"] * n,
        "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
        "Pz1": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, pz1_candidate_cheap],
    })


_BOOST_FP = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 5]
_BOOST_FP_MANTRA = [40] * 11
_BOOST_VR = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 55]
_BOOST_P1 = [50] * 11


class TestScommessaQualityBoost:
    """Gruppo Esperti bonus+media_voto as a never-negative boost on the
    SCOMMESSA gap — never a block, never a penalty."""

    def test_boost_tips_marginal_gap_over_threshold(self, cfg):
        df = _boost_pool_df()
        df["expert_bonus"] = [None] * 10 + [10]
        df["expert_media_voto"] = [None] * 10 + [10]
        result = _rend(df, fp=_BOOST_FP, fp_mantra=_BOOST_FP_MANTRA, vr=_BOOST_VR, p1=_BOOST_P1, cfg=cfg)
        assert result.iloc[-1] == "SCOMMESSA"

    def test_same_gap_without_expert_quality_data_is_not_scommessa(self, cfg):
        """Regression: identical gap (10, below SCOMMESSA_GAP_MIN=15), no
        expert quality columns at all -> unchanged pre-Gruppo-Esperti
        behavior, no SCOMMESSA."""
        df = _boost_pool_df()
        result = _rend(df, fp=_BOOST_FP, fp_mantra=_BOOST_FP_MANTRA, vr=_BOOST_VR, p1=_BOOST_P1, cfg=cfg)
        assert result.iloc[-1] != "SCOMMESSA"

    def test_boost_never_negative_when_quality_is_poor(self, cfg):
        """Quality well below SCOMMESSA_QUALITY_NEUTRAL must clip to a 0
        boost, never a penalty that makes the gap worse than ungated."""
        df = _boost_pool_df()
        df["expert_bonus"] = [None] * 10 + [1]
        df["expert_media_voto"] = [None] * 10 + [1]
        result = _rend(df, fp=_BOOST_FP, fp_mantra=_BOOST_FP_MANTRA, vr=_BOOST_VR, p1=_BOOST_P1, cfg=cfg)
        assert result.iloc[-1] != "SCOMMESSA"

    def test_exposed_gap_reflects_boost(self, cfg):
        """Fase7_Rendimento_Gap (returned gap) must reflect the boosted
        value, not the raw one — otherwise a labeled SCOMMESSA could show a
        sub-threshold gap in the frontend star rating."""
        df = _boost_pool_df()
        df["expert_bonus"] = [None] * 10 + [10]
        df["expert_media_voto"] = [None] * 10 + [10]
        label_rend, _, gap_rend, *_ = _classify(
            df, fp=_BOOST_FP, fp_mantra=_BOOST_FP_MANTRA, vr=_BOOST_VR, p1=_BOOST_P1, cfg=cfg,
        )
        assert label_rend.iloc[-1] == "SCOMMESSA"
        assert gap_rend.iloc[-1] > cfg.SCOMMESSA_GAP_MIN


class TestPrezzoExpertBlend:
    """Gruppo Esperti TOTALE blended into the Prezzo/Valore quality anchor
    alongside FP_Mantra."""

    @staticmethod
    def _pool(with_totale: bool) -> pd.DataFrame:
        n = 5
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
            "Pz1": [10, 20, 45, 40, 50],  # candidate (idx 2) -> price_pct 75
        })
        if with_totale:
            df["expert_totale"] = [10, None, 46, None, None]  # candidate -> pct 100
        return df

    def test_marginal_sopravalutato_without_expert_totale(self, cfg):
        df = self._pool(with_totale=False)
        fp_mantra = [10, 20, 50, 80, 90]  # candidate (idx 2) -> fp_mantra_pct 50
        _, _, _, label_prezzo, _, gap_prezzo = _classify(
            df, fp=[40] * 5, fp_mantra=fp_mantra, vr=[90] * 5, p1=[50] * 5, cfg=cfg,
        )
        assert label_prezzo.iloc[2] == "SOPRAVALUTATO"
        assert gap_prezzo.iloc[2] == pytest.approx(25.0)

    def test_expert_totale_rescues_marginal_sopravalutato(self, cfg):
        """Same gap as above, but the candidate has a strong Gruppo Esperti
        TOTALE — the blend must bring the gap back inside the GIUSTO band.
        With the default PREZZO_EXPERT_TOTALE_WEIGHT=0.5 (equal weight with
        FP_Mantra), quality_pct = 0.5*50 + 0.5*100 = 75 = price_pct -> gap 0."""
        df = self._pool(with_totale=True)
        fp_mantra = [10, 20, 50, 80, 90]
        _, _, _, label_prezzo, _, gap_prezzo = _classify(
            df, fp=[40] * 5, fp_mantra=fp_mantra, vr=[90] * 5, p1=[50] * 5, cfg=cfg,
        )
        assert label_prezzo.iloc[2] == "GIUSTO"
        assert gap_prezzo.iloc[2] == pytest.approx(0.0)

    def test_missing_totale_for_this_player_is_a_pure_fp_mantra_fallback(self, cfg):
        """Even when other players in the pool have expert_totale, a player
        without it must fall back to FP_Mantra-only (no blend applied to
        him specifically)."""
        df = self._pool(with_totale=True)
        # Player idx 3 has no expert_totale (already None in the fixture) —
        # his gap must match a pure FP_Mantra-vs-price calculation.
        fp_mantra = [10, 20, 50, 55, 90]  # idx 3 -> fp_mantra_pct 75 (rank4/5)
        _, _, _, _, _, gap_prezzo = _classify(
            df, fp=[40] * 5, fp_mantra=fp_mantra, vr=[90] * 5, p1=[50] * 5, cfg=cfg,
        )
        # Pz1 idx3 = 40 -> price_pct 50 (rank3/5); fp_mantra_pct 75 -> gap = -25
        assert gap_prezzo.iloc[3] == pytest.approx(-25.0)
