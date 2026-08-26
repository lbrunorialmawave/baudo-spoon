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
        within the role pool, not two independent absolute thresholds."""
        n = 6
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n, "Pr": [0.0] * n, "DV": [5.0] * n,
        })
        # Candidate (last row): lowest raw FP, highest VR in the pool.
        fp = [80, 78, 76, 74, 72, 20]
        fp_mantra = [40] * n  # keep everyone well below TOP_FP_SOGLIA (80)
        vr = [90, 92, 94, 96, 98, 130]
        p1 = [50] * n
        result = _rend(df, fp=fp, fp_mantra=fp_mantra, vr=vr, p1=p1, cfg=cfg)
        assert result.iloc[-1] == "SCOMMESSA"

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
        df = pd.DataFrame({
            "ruolo_primario": ["C", "C"],
            "Stagioni_IT": [0, 0], "Pr": [0.0, 0.0], "DV": [5.0, 5.0],
            "Pz1": [10, 100],
        })
        _, _, _, label_prezzo, _, gap_prezzo = _classify(
            df, fp=[40, 40], fp_mantra=[40, 40], vr=[130, 40], p1=[50, 50], cfg=cfg,
        )
        assert label_prezzo.iloc[0] == "AFFARE"  # cheap, high VR
        assert label_prezzo.iloc[1] == "SOPRAVALUTATO"  # expensive, low VR
        assert gap_prezzo.iloc[0] < 0
        assert gap_prezzo.iloc[1] > 0

    def test_giusto_when_price_tracks_value(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * 3,
            "Stagioni_IT": [0] * 3, "Pr": [0.0] * 3, "DV": [5.0] * 3,
            "Pz1": [10, 50, 90],
        })
        # Price and VR rank in the same order -> gap ~ 0 for all three.
        _, _, _, label_prezzo, _, _ = _classify(
            df, fp=[40] * 3, fp_mantra=[40] * 3, vr=[60, 100, 140], p1=[50] * 3, cfg=cfg,
        )
        assert (label_prezzo == "GIUSTO").all()

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
