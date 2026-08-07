"""Tests for Fase 7 classification (ml/mantra/fase7.py::classify_fase7).

Covers each label in isolation, the rule-priority order (rules are
evaluated TOP > AFFARE > SCOMMESSA > CERTEZZA > SOPRAVALUTATO > GIUSTO,
first match wins), the percentile-per-role-pool threshold mode (and its
small-pool fallback to fixed absolute thresholds), and the Fase7_Motivo
"why no classification" explanation.
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

    A single role with n < SOGLIA_POOL (20) players — small enough that
    percentile mode falls back to the fixed absolute thresholds, so these
    fixtures behave identically regardless of cfg.FASE7_THRESHOLD_MODE.
    """
    return pd.DataFrame({
        "ruolo_primario": [ruolo] * n,
        "Stagioni_IT": [0] * n,
        "Pr": [0.0] * n,
        "DV": [5.0] * n,
    })


def _classify(df, fp, fp_mantra, vr, p1, cfg):
    """Classify and return only the label Series (drop Fase7_Motivo) —
    convenience wrapper for tests that don't care about the reason text."""
    label, _motivo = classify_fase7(
        df,
        pd.Series(fp, index=df.index),
        pd.Series(fp_mantra, index=df.index),
        pd.Series(vr, index=df.index),
        pd.Series(p1, index=df.index),
        cfg,
    )
    return label


class TestIndividualLabels:
    def test_top(self, cfg):
        df = _base_df(1)
        # VR must clear TOP_VR_SOGLIA (100) in absolute/small-pool mode
        result = _classify(df, fp=[70], fp_mantra=[85], vr=[101], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_affare(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[65], fp_mantra=[65], vr=[150], p1=[50], cfg=cfg)
        assert result.iloc[0] == "AFFARE"

    def test_scommessa(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[40], fp_mantra=[40], vr=[140], p1=[50], cfg=cfg)
        assert result.iloc[0] == "SCOMMESSA"

    def test_certezza(self, cfg):
        """CERTEZZA requires Stagioni_IT>=2, Pr>=0.70, DV<=role-pool median, P1>=55.

        Before the player_season_aggregates SQL fix, Stagioni_IT and Pr could
        never reach these values on real data (Stagioni_IT was hardcoded to 1
        by a GROUP BY bug, and Pr measured stat-category coverage instead of
        playing time) — CERTEZZA was structurally unreachable. This test
        exercises the rule in isolation, independent of the data source.
        """
        df = pd.DataFrame({
            "ruolo_primario": ["C", "C", "C"],
            "Stagioni_IT": [2, 2, 2],
            "Pr": [0.80, 0.75, 0.85],
            "DV": [3.0, 3.0, 3.0],
        })
        # FP/VR kept in "neutral" ranges so no earlier rule (TOP/AFFARE/SCOMMESSA)
        # pre-empts CERTEZZA for these rows.
        result = _classify(
            df, fp=[55, 55, 55], fp_mantra=[55, 55, 55], vr=[95, 95, 95],
            p1=[60, 58, 65], cfg=cfg,
        )
        assert (result == "CERTEZZA").all()

    def test_certezza_fails_below_stagioni_threshold(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"],
            "Stagioni_IT": [1],  # below CERTEZZA_STAGIONI=2
            "Pr": [0.80],
            "DV": [3.0],
        })
        result = _classify(df, fp=[55], fp_mantra=[55], vr=[95], p1=[60], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_certezza_fails_below_pr_threshold(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"],
            "Stagioni_IT": [2],
            "Pr": [0.50],  # below CERTEZZA_PR=0.70
            "DV": [3.0],
        })
        result = _classify(df, fp=[55], fp_mantra=[55], vr=[95], p1=[60], cfg=cfg)
        assert result.iloc[0] != "CERTEZZA"

    def test_sopravalutato(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[40], fp_mantra=[40], vr=[70], p1=[50], cfg=cfg)
        assert result.iloc[0] == "SOPRAVALUTATO"

    def test_giusto(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[40], fp_mantra=[40], vr=[100], p1=[50], cfg=cfg)
        assert result.iloc[0] == "GIUSTO"

    def test_none_when_no_rule_matches(self, cfg):
        df = _base_df(1)
        # VR=85 is between SOPRAVALUTATO_VR (80) and GIUSTO_VR_MIN (90): no match.
        result = _classify(df, fp=[40], fp_mantra=[40], vr=[85], p1=[50], cfg=cfg)
        assert pd.isna(result.iloc[0])


class TestRulePriority:
    def test_top_wins_over_certezza(self, cfg):
        """A player qualifying for both TOP and CERTEZZA must be labeled TOP,
        since TOP is evaluated first in _LABEL_ORDER."""
        df = pd.DataFrame({
            "ruolo_primario": ["C"],
            "Stagioni_IT": [2],
            "Pr": [0.90],
            "DV": [1.0],
        })
        result = _classify(
            df, fp=[85], fp_mantra=[85], vr=[101], p1=[90], cfg=cfg,
        )
        assert result.iloc[0] == "TOP"

    def test_scommessa_wins_over_certezza(self, cfg):
        """SCOMMESSA is evaluated before CERTEZZA."""
        df = pd.DataFrame({
            "ruolo_primario": ["C"],
            "Stagioni_IT": [2],
            "Pr": [0.90],
            "DV": [1.0],
        })
        result = _classify(
            df, fp=[40], fp_mantra=[40], vr=[140], p1=[90], cfg=cfg,
        )
        assert result.iloc[0] == "SCOMMESSA"


class TestPercentileMode:
    def test_default_mode_is_percentile(self, cfg):
        assert cfg.FASE7_THRESHOLD_MODE == "percentile"

    def test_top_is_role_relative_not_global(self, cfg):
        """A role pool whose FP_Mantra never reaches the fixed absolute TOP
        threshold (80) still produces a TOP label in percentile mode,
        because the pool is large enough (>= SOGLIA_POOL) for TOP to be
        evaluated relative to the role's own distribution rather than a
        single global number that would systematically starve this role.

        TOP also requires VR above the role-pool VR floor (TOP_VR_PERCENTILE),
        so the top-FP player is given a VR in the upper half of the pool.
        """
        n = 25  # >= SOGLIA_POOL (20) -> percentile mode applies, no small-pool fallback
        fp_mantra_values = list(np.linspace(30.0, 54.0, n))  # all well below 80
        df = pd.DataFrame({
            "ruolo_primario": ["C"] * n,
            "Stagioni_IT": [0] * n,
            "Pr": [0.0] * n,
            "DV": [5.0] * n,
        })
        vr_values = list(np.linspace(80.0, 130.0, n))
        vr = pd.Series(vr_values, index=df.index)
        p1 = pd.Series([50.0] * n, index=df.index)
        fp_mantra = pd.Series(fp_mantra_values, index=df.index)

        label, _ = classify_fase7(df, fp_mantra, fp_mantra, vr, p1, cfg)
        assert label.iloc[-1] == "TOP"  # highest FP_Mantra + high VR in the pool

        abs_cfg = replace(cfg, FASE7_THRESHOLD_MODE="absolute")
        label_abs, _ = classify_fase7(df, fp_mantra, fp_mantra, vr, p1, abs_cfg)
        assert (label_abs == "TOP").sum() == 0  # nobody ever reaches the fixed 80 threshold

    def test_small_pool_falls_back_to_absolute(self, cfg):
        """Without the small-pool fallback, a lone player's own P85 quantile
        equals their own score, so `score > quantile` is false for everyone
        in a pool of 1 — TOP would be structurally unreachable. The
        SOGLIA_POOL fallback avoids this by using the fixed absolute
        threshold instead, so a single "C" player with FP_Mantra=85 is
        still classified TOP under the default percentile mode."""
        df = _base_df(1)
        # VR must exceed TOP_VR_SOGLIA (100) absolute fallback
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[101], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"


class TestFase7Motivo:
    def test_none_for_a_classified_player(self, cfg):
        df = _base_df(1)
        label, motivo = classify_fase7(
            df,
            pd.Series([70], index=df.index),
            pd.Series([85], index=df.index),
            pd.Series([101], index=df.index),
            pd.Series([50], index=df.index),
            cfg,
        )
        assert label.iloc[0] == "TOP"
        assert motivo.iloc[0] is None

    def test_zona_neutra_between_sopravalutato_and_giusto(self, cfg):
        df = _base_df(1)
        # VR=85: at/above SOPRAVALUTATO_VR(80) but below GIUSTO_VR_MIN(90).
        _label, motivo = classify_fase7(
            df,
            pd.Series([40], index=df.index),
            pd.Series([40], index=df.index),
            pd.Series([85], index=df.index),
            pd.Series([50], index=df.index),
            cfg,
        )
        assert "zona neutra" in motivo.iloc[0]
        assert "85" in motivo.iloc[0]

    def test_above_giusto_but_not_affare_or_top(self, cfg):
        df = _base_df(1)
        # VR=115: above GIUSTO_VR_MAX(110), but FP_Mantra(40) too low for AFFARE/TOP.
        _label, motivo = classify_fase7(
            df,
            pd.Series([40], index=df.index),
            pd.Series([40], index=df.index),
            pd.Series([115], index=df.index),
            pd.Series([50], index=df.index),
            cfg,
        )
        assert "AFFARE" in motivo.iloc[0]
        assert "TOP" in motivo.iloc[0]

    def test_quasi_certezza_names_the_single_failing_condition(self, cfg):
        df = pd.DataFrame({
            "ruolo_primario": ["C"],
            "Stagioni_IT": [2],   # passes (>=2)
            "Pr": [0.62],         # fails: below CERTEZZA_PR (0.70)
            "DV": [3.0],          # passes (pool-of-1 median == itself)
        })
        _label, motivo = classify_fase7(
            df,
            pd.Series([40], index=df.index),
            pd.Series([40], index=df.index),
            pd.Series([85], index=df.index),  # neutral VR zone
            pd.Series([60], index=df.index),  # passes (>=55)
            cfg,
        )
        assert "Quasi CERTEZZA" in motivo.iloc[0]
        assert "Pr" in motivo.iloc[0]


class TestTopExternalGates:
    """TOP blocked by present ML / expert signals; nulls never block."""

    def test_expert_rating_blocks_top(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [4.0]  # below TOP_EXPERT_MIN (6)
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_expert_rating_passes_top(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [7.0]
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_missing_expert_does_not_block(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_ml_below_soglia_blocks_top(self, cfg):
        df = _base_df(1)
        df["ruolo_primario"] = ["Por"]
        df["predicted_next_fantavoto"] = [5.0]  # Por soglia 5.7
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_ml_above_soglia_allows_top(self, cfg):
        df = _base_df(1)
        df["ruolo_primario"] = ["Por"]
        df["predicted_next_fantavoto"] = [6.0]
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[120], p1=[50], cfg=cfg)
        assert result.iloc[0] == "TOP"

    def test_vr_below_floor_blocks_top(self, cfg):
        df = _base_df(1)
        result = _classify(df, fp=[85], fp_mantra=[85], vr=[90], p1=[50], cfg=cfg)
        assert result.iloc[0] != "TOP"

    def test_motivo_mentions_expert_block(self, cfg):
        df = _base_df(1)
        df["expert_rating"] = [4.0]
        label, motivo = classify_fase7(
            df,
            pd.Series([85], index=df.index),
            pd.Series([85], index=df.index),
            pd.Series([120], index=df.index),
            pd.Series([50], index=df.index),
            cfg,
        )
        assert label.iloc[0] != "TOP"
        assert motivo.iloc[0] is not None
        assert "esperti" in motivo.iloc[0].lower() or "rating" in motivo.iloc[0].lower()
