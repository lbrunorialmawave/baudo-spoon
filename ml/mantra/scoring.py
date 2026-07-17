"""FP, VR, Prezzo Massimo — final scoring pipeline.

Steps
-----
1. FP = P1*w1 + P2*w2 + P3*w3 + P4*w4
2. FP_std = (FP - mean_FP_pool) / std_FP_pool   (per role, extended pool)
3. k = clip(1 / %(FP_std > 1.5 in pool), 1, 6)
4. FP_Corr = clip(50 + 50 * tanh(FP_std * k / 10), 0, 100)
5. CP_std = (CP - mean_CP_pool) / std_CP_pool
6. CP_Corr = clip(50 + CP_std * 10, 5, 100)
7. Fattore_Flessibilità: 1 role → 1.00, 2 → 1.05, 3+ → 1.08
8. FP_Mantra = clip(FP_Corr * Fattore_Flessibilità, 0, 100)
9. Fattore_Eroe = clip(1 + (1 - CP / CP_medio_tutti) * 0.5, 0.6, 1.6)
10. VR = clip((FP_Mantra * Fattore_Eroe / CP_Corr) * 100, 0, 300)
11. Prezzo_Massimo = max(CP * (VR / 100), 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso


def _pool_mean_std(
    series: pd.Series,
    roles: pd.Series,
    pool_roles_map: dict[str, set[str]],
) -> tuple[pd.Series, pd.Series]:
    """Return (mean, std) of *series* for the extended pool of each role."""
    mean_s = pd.Series(np.nan, index=series.index)
    std_s = pd.Series(np.nan, index=series.index)
    for ruolo, pool_set in pool_roles_map.items():
        mask = roles.isin(pool_set)
        if mask.sum() < 2:
            mean_s[mask] = 50.0
            std_s[mask] = 15.0
        else:
            pool_vals = series[mask].dropna()
            mean_s[mask] = pool_vals.mean()
            std_s[mask] = pool_vals.std(ddof=0) if len(pool_vals) > 1 else 15.0
    return mean_s, std_s


def compute_fp(
    p1: pd.Series,
    p2: pd.Series,
    p3: pd.Series,
    p4: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """FP = weighted sum of the four pillars."""
    return (
        p1 * cfg.PESO_P1
        + p2 * cfg.PESO_P2
        + p3 * cfg.PESO_P3
        + p4 * cfg.PESO_P4
    )


def compute_fp_corr(
    fp: pd.Series,
    cp: pd.Series,
    roles: pd.Series,
    n_ruoli: pd.Series,
    cfg: MantraConfig,
) -> dict[str, pd.Series]:
    """Compute all derived scores: FP_Corr, CP_Corr, FP_Mantra, VR, Prezzo.

    Parameters
    ----------
    fp:
        Raw FP values.
    cp:
        Costo Potenziale values.
    roles:
        Primary MANTRA role per player.
    n_ruoli:
        Number of MANTRA roles per player (1, 2, 3+).
    cfg:
        MantraConfig.

    Returns
    -------
    dict with keys: fp_corr, cp_corr, fp_mantra, fattore_flessibilita,
    fattore_eroe, vr, prezzo_massimo.
    """
    # Build pool map for each role
    all_roles = roles.unique()
    pool_roles_map: dict[str, set[str]] = {
        r: calcola_pool_esteso(r) for r in all_roles
    }

    # ── FP standardisation ───────────────────────────────────────────────────
    fp_mean, fp_std = _pool_mean_std(fp, roles, pool_roles_map)
    fp_std_z = (fp - fp_mean) / fp_std.replace(0, 15.0)

    # k = 1 / %(FP_std > 1.5)
    pct_above = (fp_std_z.abs() > 1.5).mean()
    k = np.clip(1.0 / max(pct_above, 0.01), 1.0, cfg.CAP_K)

    fp_corr = 50.0 + 50.0 * np.tanh(fp_std_z * k / 10.0)
    fp_corr = fp_corr.clip(lower=0, upper=100)

    # ── CP standardisation ───────────────────────────────────────────────────
    cp_mean, cp_std = _pool_mean_std(cp, roles, pool_roles_map)
    cp_std_z = (cp - cp_mean) / cp_std.replace(0, 15.0)
    cp_corr = 50.0 + cp_std_z * 10.0
    cp_corr = cp_corr.clip(lower=5, upper=100)

    # ── Flessibilità ─────────────────────────────────────────────────────────
    fless_map = {1: cfg.FLESSIBILITA_1, 2: cfg.FLESSIBILITA_2}
    fattore_fless = n_ruoli.map(lambda x: fless_map.get(x, cfg.FLESSIBILITA_3))

    fp_mantra = (fp_corr * fattore_fless).clip(lower=0, upper=100)

    # ── Fattore Eroe ─────────────────────────────────────────────────────────
    cp_medio_tutti = cp.mean()
    if pd.isna(cp_medio_tutti) or cp_medio_tutti == 0:
        cp_medio_tutti = 50.0

    fattore_eroe = 1.0 + (1.0 - cp / cp_medio_tutti) * 0.5
    fattore_eroe = fattore_eroe.clip(lower=cfg.FATTORE_EROE_MIN, upper=cfg.FATTORE_EROE_MAX)

    # ── VR (Valore Reale) ────────────────────────────────────────────────────
    vr = (fp_mantra * fattore_eroe / cp_corr.replace(0, 1.0)) * 100.0
    vr = vr.clip(lower=0, upper=300)

    # ── Prezzo Massimo ───────────────────────────────────────────────────────
    prezzo = np.maximum(cp * (vr / 100.0), 1.0)

    return {
        "fp_corr": fp_corr,
        "cp_corr": cp_corr,
        "fp_mantra": fp_mantra,
        "fattore_flessibilita": fattore_fless,
        "fattore_eroe": fattore_eroe,
        "vr": vr,
        "prezzo_massimo": prezzo,
    }
