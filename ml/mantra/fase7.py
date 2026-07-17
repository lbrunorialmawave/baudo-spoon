"""Fase 7 — Decision rules (mutually exclusive, evaluated in order).

Order
-----
1. 🏆 TOP            FP > 80
2. 💎 AFFARE         FP > 60 AND VR > 140
3. 🔄 SCOMMESSA      FP < 50 AND VR > 130
4. ✅ CERTEZZA       Stagioni_IT >= 2 AND Pr >= 0.70 AND DV <= mediana(DV) AND P1 >= 70
5. ⚠️ SOPRAVALUTATO  VR < 80
6. ⚖️ GIUSTO         90 <= VR <= 110
7. (none)            others
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso


_LABEL_ORDER: list[str] = [
    "TOP",
    "AFFARE",
    "SCOMMESSA",
    "CERTEZZA",
    "SOPRAVALUTATO",
    "GIUSTO",
    None,  # catch-all
]


def classify_fase7(
    df: pd.DataFrame,
    fp: pd.Series,
    fp_mantra: pd.Series,
    vr: pd.Series,
    p1: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Assign a Fase 7 label to each player.

    Parameters
    ----------
    df:
        DataFrame with columns:
        - ``Stagioni_IT``  — number of seasons in Serie A
        - ``Pr``           — presence rate (0-1)
        - ``DV``           — vote std dev
        - ``ruolo_primario`` — MANTRA primary role
    fp:
        Raw FP values.
    fp_mantra:
        FP_Mantra (flexibility-adjusted).
    vr:
        Valore Reale values.
    p1:
        P1 (Solidità) values.
    cfg:
        MantraConfig with thresholds.

    Returns
    -------
    pd.Series of string labels.
    """
    # Precompute median DV per role pool
    dv_mediane: dict[str, float] = {}
    for ruolo in df["ruolo_primario"].unique():
        pool_roles = calcola_pool_esteso(ruolo)
        pool_dv = df.loc[df["ruolo_primario"].isin(pool_roles), "DV"].dropna()
        dv_mediane[ruolo] = pool_dv.median() if len(pool_dv) > 0 else 99.0

    dv_soglia = df["ruolo_primario"].map(dv_mediane)

    result = pd.Series([None] * len(df), index=df.index)

    # 1. TOP
    mask = fp_mantra > cfg.TOP_FP_SOGLIA
    result[mask] = "TOP"

    # 2. AFFARE — use fp_mantra (flexibility-adjusted) instead of raw fp
    mask = result.isna() & (fp_mantra > cfg.AFFARE_FP_SOGLIA) & (vr > cfg.AFFARE_VR_SOGLIA)
    result[mask] = "AFFARE"

    # 3. SCOMMESSA
    mask = result.isna() & (fp < cfg.SCOMMESSA_FP_SOGLIA) & (vr > cfg.SCOMMESSA_VR_SOGLIA)
    result[mask] = "SCOMMESSA"

    # 4. CERTEZZA
    stagioni = df.get("Stagioni_IT", pd.Series(0, index=df.index)).fillna(0)
    pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
    dv = df.get("DV", pd.Series(99, index=df.index)).fillna(99)
    mask = (
        result.isna()
        & (stagioni >= cfg.CERTEZZA_STAGIONI)
        & (pr >= cfg.CERTEZZA_PR)
        & (dv <= dv_soglia)
        & (p1 >= cfg.CERTEZZA_P1)
    )
    result[mask] = "CERTEZZA"

    # 5. SOPRAVALUTATO
    mask = result.isna() & (vr < cfg.SOPRAVALUTATO_VR)
    result[mask] = "SOPRAVALUTATO"

    # 6. GIUSTO
    mask = result.isna() & (vr >= cfg.GIUSTO_VR_MIN) & (vr <= cfg.GIUSTO_VR_MAX)
    result[mask] = "GIUSTO"

    return result
