"""P4 — Mercato Storico (historical auction value pillar).

Formula
-------
CP = P1 * 0.2 + P2 * 0.3 + P3 * 0.5    # Costo Potenziale

Picco  = MAX(Pz1, Pz2, Pz3)             # peak historical price
Trend  = 0 IF Pz1 == 0 ELSE (Pz3 - Pz1) / (Pz1 + 5)

Livello = (CP / CP_max_ruolo_pool) * 30         # max 30
Picco_c = clip(Picco / max(CP, 1), 0, 2) * 25   # max 50
Trend_c = clip(Trend * 20 + 50, 0, 100) * 0.2   # max 20

P4 = clip(Livello + Picco_c + Trend_c, 0, 100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso


def compute_cp(p1: pd.Series, p2: pd.Series, p3: pd.Series) -> pd.Series:
    """Compute Costo Potenziale (CP) from the three pillars.

    CP = P1 * 0.2 + P2 * 0.3 + P3 * 0.5
    """
    return p1 * 0.2 + p2 * 0.3 + p3 * 0.5


def compute_p4(
    df: pd.DataFrame,
    p1: pd.Series,
    p2: pd.Series,
    p3: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Compute P4 (Mercato Storico) for each player.

    Parameters
    ----------
    df:
        DataFrame with columns:
        - ``ruolo_primario``  — MANTRA primary role (str)
        - ``Pz1``, ``Pz2``, ``Pz3`` — auction prices last 3 years (int, 0 if never listed)
    p1, p2, p3:
        Pre-computed pillar values.

    Returns
    -------
    pd.Series of P4 values clipped to [0, 100].
    """
    work = df.copy()
    cp = compute_cp(p1, p2, p3)

    # ── Prezzo massimo di ruolo ──────────────────────────────────────────────
    # Compute CP_max per role pool (using pool fusion, gated on real sample size)
    role_counts = work["ruolo_primario"].value_counts().to_dict()
    cp_max_pool: dict[str, float] = {}
    for ruolo in work["ruolo_primario"].unique():
        pool_roles = calcola_pool_esteso(ruolo, role_counts, cfg.SOGLIA_POOL)
        pool_mask = work["ruolo_primario"].isin(pool_roles)
        pool_cp = cp[pool_mask]
        cp_max_pool[ruolo] = pool_cp.max() if len(pool_cp) > 0 else 1.0

    cp_max_ruolo = work["ruolo_primario"].map(cp_max_pool).fillna(1.0)

    # ── Picco ────────────────────────────────────────────────────────────────
    pz1 = work.get("Pz1", pd.Series(0, index=work.index)).fillna(0).astype(float)
    pz2 = work.get("Pz2", pd.Series(0, index=work.index)).fillna(0).astype(float)
    pz3 = work.get("Pz3", pd.Series(0, index=work.index)).fillna(0).astype(float)

    picco = pd.concat([pz1, pz2, pz3], axis=1).max(axis=1)

    # ── Trend ────────────────────────────────────────────────────────────────
    trend = np.where(pz1 > 0, (pz3 - pz1) / (pz1 + 5.0), 0.0)

    # ── Componenti ───────────────────────────────────────────────────────────
    livello = (cp / cp_max_ruolo.replace(0, 1.0)) * 30.0
    picco_c = np.clip(picco / np.maximum(cp, 1.0), 0, 2) * 25.0
    trend_c = np.clip(trend * 20.0 + 50.0, 0, 100) * 0.2

    p4 = livello + picco_c + trend_c
    return p4.clip(lower=0, upper=100)
