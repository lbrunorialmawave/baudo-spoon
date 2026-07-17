"""P1 — Solidità (consistency / reliability pillar).

Formula
-------
P1 = min(Min_annuo / 2700, 1) * 30
   + (V / 10) * 25
   + (1 / (1 + DV)) * 20
   + Pr * 25

Where:
    Min_annuo  — average minutes played per season
    V          — historical average vote (media voto)
    DV         — standard deviation of votes (deviazione standard voto)
    Pr         — presence rate (fraction of matches played, 0-1)

Neo-arrivi (Stagioni_IT == 0):
    Min_annuo  — 2000 if designated starter, 500 if reserve/gamble
    V, DV      — median of the role pool
    Pr         — 0.75 if starter, 0.40 if reserve
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso


def compute_p1(
    df: pd.DataFrame,
    cfg: MantraConfig,
) -> pd.Series:
    """Compute P1 (Solidità) for each player in *df*.

    Parameters
    ----------
    df:
        DataFrame with columns:
        - ``Min_annuo``   — average minutes per season (int)
        - ``V``           — mean historical vote (float, 0-10)
        - ``DV``          — vote standard deviation (float)
        - ``Pr``          — presence rate (float, 0-1)
        - ``ruolo_primario`` — MANTRA primary role (str)
        - ``is_neo_arrivo``  — bool, True if Stagioni_IT == 0
        - ``is_starter``     — bool, True if designated starter
    cfg:
        Calibrated coefficients.

    Returns
    -------
    pd.Series with P1 values clipped to [0, 100].
    """
    work = df.copy()

    # ── Handle neo-arrivi ────────────────────────────────────────────────
    neo_mask = work.get("is_neo_arrivo", pd.Series([False] * len(work)))
    if neo_mask.any():
        for ruolo in work.loc[neo_mask, "ruolo_primario"].unique():
            pool = work[~neo_mask]
            ruolo_pool = pool[pool["ruolo_primario"].isin(calcola_pool_esteso(ruolo))]
            if ruolo_pool.empty:
                continue
            median_v = ruolo_pool["V"].median()
            median_dv = ruolo_pool["DV"].median()

            mask = neo_mask & (work["ruolo_primario"] == ruolo)
            work.loc[mask, "V"] = work.loc[mask, "V"].fillna(median_v)
            work.loc[mask, "DV"] = work.loc[mask, "DV"].fillna(median_dv)

            # Pr: 0.75 if starter, 0.40 if reserve
            starter_mask = mask & work.get("is_starter", pd.Series([False] * len(work)))
            reserve_mask = mask & ~starter_mask
            work.loc[starter_mask, "Pr"] = work.loc[starter_mask, "Pr"].fillna(0.75)
            work.loc[reserve_mask, "Pr"] = work.loc[reserve_mask, "Pr"].fillna(0.40)

            # Min_annuo: 2000 if starter, 500 if reserve
            work.loc[starter_mask, "Min_annuo"] = work.loc[starter_mask, "Min_annuo"].fillna(2000)
            work.loc[reserve_mask, "Min_annuo"] = work.loc[reserve_mask, "Min_annuo"].fillna(500)

    # ── Fill remaining NaNs with role medians ─────────────────────────────
    for ruolo in work["ruolo_primario"].unique():
        mask = work["ruolo_primario"] == ruolo
        for col in ["Min_annuo", "V", "DV", "Pr"]:
            if work.loc[mask, col].isna().any():
                median_val = work.loc[mask & ~work[col].isna(), col].median()
                if pd.isna(median_val):
                    median_val = 0
                work.loc[mask, col] = work.loc[mask, col].fillna(median_val)

    # ── Compute P1 ────────────────────────────────────────────────────────
    min_term = np.minimum(work["Min_annuo"].clip(lower=0) / cfg.SOGLIA_MINUTI_MAX, 1.0) * 30
    v_term = (work["V"].clip(lower=0, upper=10) / 10.0) * 25
    dv_term = (1.0 / (1.0 + work["DV"].clip(lower=0))) * 20
    pr_term = work["Pr"].clip(lower=0, upper=1) * 25

    p1 = min_term + v_term + dv_term + pr_term
    return p1.clip(lower=0, upper=100)
