"""Fase 7 — Decision rules (mutually exclusive, evaluated in order).

Order
-----
1. 🏆 TOP            FP_Mantra above the role-pool's top percentile
2. 💎 AFFARE         FP_Mantra and VR both above their role-pool percentiles
3. 🔄 SCOMMESSA      raw FP low, VR high — both relative to the role pool
4. ✅ CERTEZZA       Stagioni_IT >= 2 AND Pr >= 0.70 AND DV <= pool quantile AND P1 >= 55
5. ⚠️ SOPRAVALUTATO  VR below the role-pool's low percentile
6. ⚖️ GIUSTO         VR within the role-pool's "fair value" percentile band
7. (none)            others — see Fase7_Motivo for why

Threshold mode
--------------
``cfg.FASE7_THRESHOLD_MODE`` controls how TOP/AFFARE/SCOMMESSA/SOPRAVALUTATO/
GIUSTO thresholds are resolved:

- ``"percentile"`` (default): thresholds are computed per role-pool
  (``calcola_pool_esteso``) so e.g. TOP means "top ~15% of FP_Mantra within
  your role", not a single global number that may systematically favor or
  disadvantage certain roles (FP/VR incorporate role-specific pillar
  coefficients). Role pools smaller than ``cfg.SOGLIA_POOL`` fall back to the
  fixed absolute thresholds below (too few players for a reliable
  percentile).
- ``"absolute"``: always use the fixed thresholds (pre-percentile behavior),
  kept as an explicit rollback knob while only a few seasons of real data are
  available.

CERTEZZA's ``Stagioni_IT``/``Pr``/``P1`` legs are always absolute — they are
objectively comparable across roles (a season played is a season played).
Only its ``DV`` leg is pool-relative (a quantile, default = median, matching
the historical behavior).
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

_THRESHOLD_KEYS = (
    "top_fp_mantra",
    "affare_fp_mantra",
    "affare_vr",
    "scommessa_fp",
    "scommessa_vr",
    "sopravalutato_vr",
    "giusto_vr_min",
    "giusto_vr_max",
    "certezza_dv",
)


def _pool_thresholds(
    df: pd.DataFrame,
    fp: pd.Series,
    fp_mantra: pd.Series,
    vr: pd.Series,
    cfg: MantraConfig,
) -> dict[str, dict[str, float]]:
    """Precompute, once per role-pool (not per row), the percentile-based
    thresholds used by TOP/AFFARE/SCOMMESSA/SOPRAVALUTATO/GIUSTO/CERTEZZA(DV).

    Pools smaller than ``cfg.SOGLIA_POOL`` fall back to the fixed absolute
    thresholds (too few players for a reliable percentile).
    """
    out: dict[str, dict[str, float]] = {}
    for ruolo in df["ruolo_primario"].dropna().unique():
        pool_mask = df["ruolo_primario"].isin(calcola_pool_esteso(ruolo))
        pool_size = int(pool_mask.sum())
        pool_dv = df.loc[pool_mask, "DV"].dropna()
        certezza_dv = (
            float(pool_dv.quantile(cfg.CERTEZZA_DV_PERCENTILE))
            if len(pool_dv) > 0 else 99.0
        )

        if pool_size < cfg.SOGLIA_POOL:
            out[ruolo] = {
                "top_fp_mantra": cfg.TOP_FP_SOGLIA,
                "affare_fp_mantra": cfg.AFFARE_FP_SOGLIA,
                "affare_vr": cfg.AFFARE_VR_SOGLIA,
                "scommessa_fp": cfg.SCOMMESSA_FP_SOGLIA,
                "scommessa_vr": cfg.SCOMMESSA_VR_SOGLIA,
                "sopravalutato_vr": cfg.SOPRAVALUTATO_VR,
                "giusto_vr_min": cfg.GIUSTO_VR_MIN,
                "giusto_vr_max": cfg.GIUSTO_VR_MAX,
                "certezza_dv": certezza_dv,
            }
            continue

        pool_fp_mantra = fp_mantra[pool_mask].dropna()
        pool_vr = vr[pool_mask].dropna()
        pool_fp = fp[pool_mask].dropna()
        out[ruolo] = {
            "top_fp_mantra": float(pool_fp_mantra.quantile(cfg.TOP_FP_PERCENTILE)),
            "affare_fp_mantra": float(pool_fp_mantra.quantile(cfg.AFFARE_FP_PERCENTILE)),
            "affare_vr": float(pool_vr.quantile(cfg.AFFARE_VR_PERCENTILE)),
            "scommessa_fp": float(pool_fp.quantile(cfg.SCOMMESSA_FP_PERCENTILE)),
            "scommessa_vr": float(pool_vr.quantile(cfg.SCOMMESSA_VR_PERCENTILE)),
            "sopravalutato_vr": float(pool_vr.quantile(cfg.SOPRAVALUTATO_VR_PERCENTILE)),
            "giusto_vr_min": float(pool_vr.quantile(cfg.GIUSTO_VR_PERCENTILE_MIN)),
            "giusto_vr_max": float(pool_vr.quantile(cfg.GIUSTO_VR_PERCENTILE_MAX)),
            "certezza_dv": certezza_dv,
        }
    return out


def _absolute_thresholds(
    df: pd.DataFrame,
    cfg: MantraConfig,
) -> pd.DataFrame:
    """Fixed thresholds (FASE7_THRESHOLD_MODE="absolute"): same value for
    every row, except CERTEZZA's DV which stays pool-relative (matching
    historical behavior — this is not new with percentile mode)."""
    fixed = {k: cfg.__getattribute__(v) for k, v in {
        "top_fp_mantra": "TOP_FP_SOGLIA",
        "affare_fp_mantra": "AFFARE_FP_SOGLIA",
        "affare_vr": "AFFARE_VR_SOGLIA",
        "scommessa_fp": "SCOMMESSA_FP_SOGLIA",
        "scommessa_vr": "SCOMMESSA_VR_SOGLIA",
        "sopravalutato_vr": "SOPRAVALUTATO_VR",
        "giusto_vr_min": "GIUSTO_VR_MIN",
        "giusto_vr_max": "GIUSTO_VR_MAX",
    }.items()}
    th = pd.DataFrame([fixed] * len(df), index=df.index)

    dv_mediane: dict[str, float] = {}
    for ruolo in df["ruolo_primario"].unique():
        pool_roles = calcola_pool_esteso(ruolo)
        pool_dv = df.loc[df["ruolo_primario"].isin(pool_roles), "DV"].dropna()
        dv_mediane[ruolo] = (
            float(pool_dv.quantile(cfg.CERTEZZA_DV_PERCENTILE))
            if len(pool_dv) > 0 else 99.0
        )
    th["certezza_dv"] = df["ruolo_primario"].map(dv_mediane)
    return th


def _resolve_thresholds(
    df: pd.DataFrame,
    fp: pd.Series,
    fp_mantra: pd.Series,
    vr: pd.Series,
    cfg: MantraConfig,
) -> pd.DataFrame:
    """Return a per-row DataFrame with one column per threshold in
    ``_THRESHOLD_KEYS``, resolved according to ``cfg.FASE7_THRESHOLD_MODE``."""
    if cfg.FASE7_THRESHOLD_MODE == "absolute":
        return _absolute_thresholds(df, cfg)

    pool_th = _pool_thresholds(df, fp, fp_mantra, vr, cfg)
    default = {k: np.nan for k in _THRESHOLD_KEYS}
    rows = [pool_th.get(ruolo, default) for ruolo in df["ruolo_primario"]]
    th = pd.DataFrame(rows, index=df.index)
    return th


def _explain_unclassified(
    df: pd.DataFrame,
    fp_mantra: pd.Series,
    vr: pd.Series,
    p1: pd.Series,
    th: pd.DataFrame,
    result: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Explain, for players with no Fase7 label, which rule(s) they came
    closest to matching. Computed only for the unlabeled subset (typically
    ~100-170 players/season) — not worth vectorizing.
    """
    motivo = pd.Series([None] * len(df), index=df.index, dtype=object)
    stagioni = df.get("Stagioni_IT", pd.Series(0, index=df.index)).fillna(0)
    pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
    dv = df.get("DV", pd.Series(99, index=df.index)).fillna(99)

    for idx in df.index[result.isna()]:
        conditions = {
            "Stagioni_IT": (
                stagioni.at[idx] >= cfg.CERTEZZA_STAGIONI,
                f"Stagioni_IT {stagioni.at[idx]:.0f} < {cfg.CERTEZZA_STAGIONI}",
            ),
            "Pr": (
                pr.at[idx] >= cfg.CERTEZZA_PR,
                f"Pr {pr.at[idx]:.2f} < {cfg.CERTEZZA_PR:.2f}",
            ),
            "DV": (
                dv.at[idx] <= th.at[idx, "certezza_dv"],
                f"DV {dv.at[idx]:.1f} > {th.at[idx, 'certezza_dv']:.1f}",
            ),
            "P1": (
                p1.at[idx] >= cfg.CERTEZZA_P1,
                f"P1 {p1.at[idx]:.0f} < {cfg.CERTEZZA_P1:.0f}",
            ),
        }
        failing = [msg for ok, msg in conditions.values() if not ok]
        if len(failing) == 1:
            motivo.at[idx] = f"Quasi CERTEZZA: manca solo {failing[0]}"
            continue

        v = vr.at[idx]
        sopra = th.at[idx, "sopravalutato_vr"]
        gmin = th.at[idx, "giusto_vr_min"]
        gmax = th.at[idx, "giusto_vr_max"]
        if v > gmax:
            top_th = th.at[idx, "top_fp_mantra"]
            affare_fp_th = th.at[idx, "affare_fp_mantra"]
            motivo.at[idx] = (
                f"VR {v:.0f} sopra GIUSTO (>{gmax:.0f}) ma FP_Mantra "
                f"{fp_mantra.at[idx]:.0f} non basta per AFFARE "
                f"(serve >{affare_fp_th:.0f}) o TOP (serve >{top_th:.0f})"
            )
        else:
            motivo.at[idx] = (
                f"VR {v:.0f}: tra SOPRAVALUTATO (<{sopra:.0f}) e GIUSTO "
                f"({gmin:.0f}-{gmax:.0f}): zona neutra"
            )
    return motivo


def classify_fase7(
    df: pd.DataFrame,
    fp: pd.Series,
    fp_mantra: pd.Series,
    vr: pd.Series,
    p1: pd.Series,
    cfg: MantraConfig,
) -> tuple[pd.Series, pd.Series]:
    """Assign a Fase 7 label (and, for unlabeled players, a reason) to each player.

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
        MantraConfig with thresholds (see ``FASE7_THRESHOLD_MODE``).

    Returns
    -------
    ``(label, motivo)`` — two ``pd.Series`` aligned to ``df.index``.
    ``motivo`` is populated only for rows where ``label`` is ``None``.
    """
    th = _resolve_thresholds(df, fp, fp_mantra, vr, cfg)

    result = pd.Series([None] * len(df), index=df.index, dtype=object)

    # 1. TOP
    mask = fp_mantra > th["top_fp_mantra"]
    result[mask] = "TOP"

    # 2. AFFARE — use fp_mantra (flexibility-adjusted) instead of raw fp
    mask = result.isna() & (fp_mantra > th["affare_fp_mantra"]) & (vr > th["affare_vr"])
    result[mask] = "AFFARE"

    # 3. SCOMMESSA
    mask = result.isna() & (fp < th["scommessa_fp"]) & (vr > th["scommessa_vr"])
    result[mask] = "SCOMMESSA"

    # 4. CERTEZZA
    stagioni = df.get("Stagioni_IT", pd.Series(0, index=df.index)).fillna(0)
    pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
    dv = df.get("DV", pd.Series(99, index=df.index)).fillna(99)
    mask = (
        result.isna()
        & (stagioni >= cfg.CERTEZZA_STAGIONI)
        & (pr >= cfg.CERTEZZA_PR)
        & (dv <= th["certezza_dv"])
        & (p1 >= cfg.CERTEZZA_P1)
    )
    result[mask] = "CERTEZZA"

    # 5. SOPRAVALUTATO
    mask = result.isna() & (vr < th["sopravalutato_vr"])
    result[mask] = "SOPRAVALUTATO"

    # 6. GIUSTO
    mask = result.isna() & (vr >= th["giusto_vr_min"]) & (vr <= th["giusto_vr_max"])
    result[mask] = "GIUSTO"

    motivo = _explain_unclassified(df, fp_mantra, vr, p1, th, result, cfg)

    return result, motivo
