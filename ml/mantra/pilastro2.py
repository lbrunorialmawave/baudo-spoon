"""P2 — Potenziale (upside / ceiling pillar).

Formula
-------
Pool statistico = only players with Min_annuo >= 450 (in role / fused group)

If Min_annuo < 450:
    z_qualita = 0, z_output = 0
Else:
    z_qualita = zscore(xG90 + xA90, pool)
    z_output  = zscore(G90 + A90, pool)

P2 = clip(50 + (z_qualita * 0.60 + z_output * 0.40) * 15, 0, 100)

Portieri (P2bis):
    z_parate = zscore(saves_per90, pool)
    z_clean  = zscore(clean_sheet_per90, pool)
    z_uscite = zscore(claims_per90, pool)
    P2 = clip(50 + (z_parate*0.50 + z_clean*0.35 + z_uscite*0.15) * 15, 0, 100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso


def _zscore(series: pd.Series, pool: pd.Series) -> pd.Series:
    """Compute z-score of *series* against the *pool* distribution."""
    mu = pool.mean()
    sigma = pool.std(ddof=0)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def compute_p2(
    df: pd.DataFrame,
    cfg: MantraConfig,
) -> pd.Series:
    """Compute P2 (Potenziale) for each player.

    Parameters
    ----------
    df:
        DataFrame with columns:
        - ``Min_annuo``       — average minutes (int)
        - ``ruolo_primario``  — MANTRA primary role (str)
        - ``xG90``, ``xA90``  — expected goals/assists per 90 (float)
        - ``G90``, ``A90``    — actual goals/assists per 90 (float)
        - ``saves_per90``     — saves per 90 (GK only)
        - ``clean_sheet_per90`` — clean sheets per 90 (GK/DEF only)
        - ``claims_per90``    — claims/crosses caught per 90 (GK only, optional)
    cfg:
        Calibrated coefficients.

    Returns
    -------
    pd.Series with P2 values clipped to [0, 100].
    """
    work = df.copy()
    soglia = cfg.SOGLIA_MINUTI_MIN  # 450
    above_threshold = work["Min_annuo"].fillna(0) >= soglia

    result = pd.Series(50.0, index=work.index)  # default = neutral

    # ── Portieri ──────────────────────────────────────────────────────────
    gk_mask = work["ruolo_primario"] == "Por"
    if gk_mask.any():
        gk_pool_mask = gk_mask & above_threshold
        gk_pool = work.loc[gk_pool_mask]

        if len(gk_pool) > 0:
            z_parate = _zscore(
                work["saves_per90"].fillna(0),
                gk_pool["saves_per90"],
            )
            z_clean = _zscore(
                work["clean_sheet_per90"].fillna(0),
                gk_pool["clean_sheet_per90"],
            )
            z_uscite = _zscore(
                work.get("claims_per90", pd.Series(0, index=work.index)).fillna(0),
                gk_pool.get("claims_per90", pd.Series(0, index=gk_pool.index)),
            )

            gk_score = (
                z_parate * 0.50
                + z_clean * 0.35
                + z_uscite * 0.15
            )
            # Zero out below-threshold players
            gk_score = gk_score.where(above_threshold, 0.0)
            result = 50 + gk_score * 15
        else:
            # No GK pool above threshold, all get neutral score
            result = pd.Series(50.0, index=work.index)

    # ── Outfield players ──────────────────────────────────────────────────
    outfield_mask = work["ruolo_primario"] != "Por"
    if outfield_mask.any():
        # Build role-specific pool (with pool fusion)
        for ruolo in work.loc[outfield_mask, "ruolo_primario"].unique():
            pool_roles = calcola_pool_esteso(ruolo)
            mask = (work["ruolo_primario"] == ruolo)

            # Statistical pool: same/fused role + above threshold
            pool_mask = (
                work["ruolo_primario"].isin(pool_roles)
                & above_threshold
            )
            pool = work.loc[pool_mask]

            if len(pool) < 2:
                # Not enough data, assign neutral
                result[mask] = 50.0
                continue

            # Quality z-score (xG90 + xA90)
            qualita = work["xG90"].fillna(0) + work["xA90"].fillna(0)
            pool_qualita = pool["xG90"].fillna(0) + pool["xA90"].fillna(0)
            z_qualita = _zscore(qualita, pool_qualita)

            # Output z-score (G90 + A90)
            output = work["G90"].fillna(0) + work["A90"].fillna(0)
            pool_output = pool["G90"].fillna(0) + pool["A90"].fillna(0)
            z_output = _zscore(output, pool_output)

            # Combined score
            combined = z_qualita * 0.60 + z_output * 0.40
            # Zero out below-threshold players in this role
            combined = combined.where(
                mask & above_threshold,
                combined.where(~mask, 0.0),
            )
            result[mask] = 50 + combined[mask] * 15

    return result.clip(lower=0, upper=100)
