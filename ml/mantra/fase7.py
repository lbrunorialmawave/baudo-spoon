"""Fase 7 — two independent decision axes.

Axis 1: Rendimento/Affidabilità (mutually exclusive, first match wins)
-----------------------------------------------------------------------
1. 🏆 TOP        FP_Mantra above the role-pool's top percentile
                 AND VR above the role-pool VR floor (median-ish)
                 AND (optional) ML next-fantavoto ≥ role soglia when present
                 AND (optional) expert rating ≥ TOP_EXPERT_MIN when present
2. ✅ CERTEZZA   Stagioni_IT >= 2 AND Pr >= 0.70 AND DV <= pool quantile AND
                 P1 >= 55 ... OR titolarita_attesa (EWMA of matchday-status
                 probability) >= CERTEZZA_TITOLARITA_SOGLIA AND DV <= pool
                 quantile — a forward-looking leg for players who lack
                 Serie A history but have a locked-in starting spot.
3. 🔄 SCOMMESSA  percentile(VR) - percentile(FP) > SCOMMESSA_GAP_MIN within
                 the role pool — VR sees more upside than raw output shows.
4. (none)        see Fase7_Rendimento_Motivo for why.

Axis 2: Prezzo/Valore (three contiguous bands of one gap)
-----------------------------------------------------------------------
Let ``gap = percentile(quotation) - percentile(FP_Mantra)`` within the role
pool. Deliberately FP_Mantra, not VR: VR already bakes in an anti-cost
adjustment (``Fattore_Eroe`` discounts VR for high-CP/expensive players, to
reward cheap sleepers), so comparing quotation against VR double-counts
price — an elite AND expensive player (high quotation percentile) gets a
*dampened* VR percentile by design, which would mislabel him SOPRAVALUTATO
even though his price is fully justified by his output. FP_Mantra carries
no such cost adjustment, so the gap here answers "is the price justified by
how good he actually is", not "is he a bargain relative to an already
cost-adjusted score".
- 💎 AFFARE          gap <= -GIUSTO_GAP_BAND   (cheap relative to quality)
- ⚖️ GIUSTO          |gap| < GIUSTO_GAP_BAND    (price tracks quality)
- ⚠️ SOPRAVALUTATO   gap >= GIUSTO_GAP_BAND    (pricey relative to quality)
- (none)             no quotation available — see Fase7_Prezzo_Motivo.

A player can carry a label on both axes independently (e.g. CERTEZZA +
AFFARE), one, or neither — the two axes never compete with each other.

Threshold mode
--------------
``cfg.FASE7_THRESHOLD_MODE`` controls how TOP/CERTEZZA(DV) thresholds are
resolved (the only two legs left that use a per-role threshold rather than
a pool-wide percentile gap):

- ``"percentile"`` (default): thresholds are computed per role-pool
  (``calcola_pool_esteso``) so e.g. TOP means "top ~10% of FP_Mantra within
  your role", not a single global number that may systematically favor or
  disadvantage certain roles. Role pools smaller than ``cfg.SOGLIA_POOL``
  fall back to the fixed absolute thresholds below.
- ``"absolute"``: always use the fixed thresholds (pre-percentile
  behavior), kept as an explicit rollback knob.

CERTEZZA's ``Stagioni_IT``/``Pr``/``P1`` legs are always absolute — they are
objectively comparable across roles. Only its ``DV`` leg is pool-relative
(a quantile, default = median).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.roles import calcola_pool_esteso
from ml.mantra.scoring import _pool_percentile

_THRESHOLD_KEYS = ("top_fp_mantra", "top_vr", "certezza_dv")


def _pool_thresholds(
    df: pd.DataFrame,
    fp_mantra: pd.Series,
    vr: pd.Series,
    cfg: MantraConfig,
) -> dict[str, dict[str, float]]:
    """Precompute, once per role-pool, the percentile thresholds used by
    TOP and CERTEZZA's DV leg — the only legs still threshold-based.

    Pools smaller than ``cfg.SOGLIA_POOL`` fall back to the fixed absolute
    thresholds (too few players for a reliable percentile).
    """
    out: dict[str, dict[str, float]] = {}
    role_counts = df["ruolo_primario"].value_counts().to_dict()
    for ruolo in df["ruolo_primario"].dropna().unique():
        pool_roles = calcola_pool_esteso(ruolo, role_counts, cfg.SOGLIA_POOL)
        pool_mask = df["ruolo_primario"].isin(pool_roles)
        pool_size = int(pool_mask.sum())
        pool_dv = df.loc[pool_mask, "DV"].dropna()
        certezza_dv = (
            float(pool_dv.quantile(cfg.CERTEZZA_DV_PERCENTILE))
            if len(pool_dv) > 0 else 99.0
        )

        if pool_size < cfg.SOGLIA_POOL:
            out[ruolo] = {
                "top_fp_mantra": cfg.TOP_FP_SOGLIA,
                "top_vr": cfg.TOP_VR_SOGLIA,
                "certezza_dv": certezza_dv,
            }
            continue

        pool_fp_mantra = fp_mantra[pool_mask].dropna()
        pool_vr = vr[pool_mask].dropna()
        out[ruolo] = {
            "top_fp_mantra": float(pool_fp_mantra.quantile(cfg.TOP_FP_PERCENTILE)),
            "top_vr": float(pool_vr.quantile(cfg.TOP_VR_PERCENTILE)),
            "certezza_dv": certezza_dv,
        }
    return out


def _absolute_thresholds(df: pd.DataFrame, cfg: MantraConfig) -> pd.DataFrame:
    """Fixed thresholds (FASE7_THRESHOLD_MODE="absolute"): same value for
    every row, except CERTEZZA's DV which stays pool-relative."""
    fixed = {"top_fp_mantra": cfg.TOP_FP_SOGLIA, "top_vr": cfg.TOP_VR_SOGLIA}
    th = pd.DataFrame([fixed] * len(df), index=df.index)

    dv_mediane: dict[str, float] = {}
    role_counts = df["ruolo_primario"].value_counts().to_dict()
    for ruolo in df["ruolo_primario"].unique():
        pool_roles = calcola_pool_esteso(ruolo, role_counts, cfg.SOGLIA_POOL)
        pool_dv = df.loc[df["ruolo_primario"].isin(pool_roles), "DV"].dropna()
        dv_mediane[ruolo] = (
            float(pool_dv.quantile(cfg.CERTEZZA_DV_PERCENTILE))
            if len(pool_dv) > 0 else 99.0
        )
    th["certezza_dv"] = df["ruolo_primario"].map(dv_mediane)
    return th


def _resolve_thresholds(
    df: pd.DataFrame,
    fp_mantra: pd.Series,
    vr: pd.Series,
    cfg: MantraConfig,
) -> pd.DataFrame:
    if cfg.FASE7_THRESHOLD_MODE == "absolute":
        return _absolute_thresholds(df, cfg)

    pool_th = _pool_thresholds(df, fp_mantra, vr, cfg)
    default = {k: np.nan for k in _THRESHOLD_KEYS}
    rows = [pool_th.get(ruolo, default) for ruolo in df["ruolo_primario"]]
    return pd.DataFrame(rows, index=df.index)


def _build_pool_map(df: pd.DataFrame, cfg: MantraConfig) -> dict[str, set[str]]:
    role_counts = df["ruolo_primario"].value_counts().to_dict()
    return {
        r: calcola_pool_esteso(r, role_counts, cfg.SOGLIA_POOL)
        for r in df["ruolo_primario"].dropna().unique()
    }


def _price_percentile(
    price: pd.Series,
    roles: pd.Series,
    pool_map: dict[str, set[str]],
) -> pd.Series:
    """Percentile (0-100) of each player's own quotation within his role
    pool, using only quoted players (price > 0) as the reference
    population. Never-quoted players get NaN — excluded from the pool
    population entirely, not given an arbitrary rank."""
    valid = price.where(price > 0)
    out = pd.Series(np.nan, index=price.index)
    for ruolo, pool_set in pool_map.items():
        mask = roles.isin(pool_set)
        pool_vals = valid[mask].dropna()
        n = len(pool_vals)
        if n <= 1:
            continue
        ranks = pool_vals.rank(method="average") - 1.0
        out[pool_vals.index] = (ranks / (n - 1)) * 100.0
    return out


def _external_ml_ok(df: pd.DataFrame, cfg: MantraConfig) -> pd.Series:
    """True where ML next-fantavoto is missing OR meets the role soglia.

    Looks for ``predicted_next_fantavoto`` first, then ``predicted_fantavoto``.
    """
    pred = None
    for col in ("predicted_next_fantavoto", "predicted_fantavoto"):
        if col in df.columns:
            pred = pd.to_numeric(df[col], errors="coerce")
            break
    if pred is None:
        return pd.Series(True, index=df.index)

    roles = df.get("ruolo_primario", pd.Series(index=df.index, dtype=object))
    soglie = cfg.NEXT_FANTAVOTO_MIN_BY_ROLE
    min_req = roles.map(lambda r: soglie.get(r, 6.0) if isinstance(r, str) else 6.0)
    return pred.isna() | (pred >= min_req)


def _external_expert_ok(df: pd.DataFrame, cfg: MantraConfig) -> pd.Series:
    """True where expert rating is missing OR >= TOP_EXPERT_MIN."""
    if "expert_rating" not in df.columns:
        return pd.Series(True, index=df.index)
    rating = pd.to_numeric(df["expert_rating"], errors="coerce")
    return rating.isna() | (rating >= cfg.TOP_EXPERT_MIN)


def _explain_rendimento(
    df: pd.DataFrame,
    fp_mantra: pd.Series,
    vr: pd.Series,
    p1: pd.Series,
    gap_rend: pd.Series,
    th: pd.DataFrame,
    label_rend: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Explain, for players with no Rendimento-axis label, which rule(s)
    they came closest to matching. Computed only for the unlabeled subset —
    not worth vectorizing."""
    motivo = pd.Series([None] * len(df), index=df.index, dtype=object)
    stagioni = df.get("Stagioni_IT", pd.Series(0, index=df.index)).fillna(0)
    pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
    dv = df.get("DV", pd.Series(99, index=df.index)).fillna(99)
    tit = df.get("titolarita_attesa", pd.Series(np.nan, index=df.index))

    for idx in df.index[label_rend.isna()]:
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
            motivo.at[idx] = f"Quasi CERTEZZA (storico): manca solo {failing[0]}"
            continue

        dv_ok = conditions["DV"][0]
        tit_val = tit.at[idx]
        if dv_ok and pd.notna(tit_val):
            mancante = cfg.CERTEZZA_TITOLARITA_SOGLIA - float(tit_val)
            if 0 < mancante <= 10:
                motivo.at[idx] = (
                    f"Quasi CERTEZZA (via titolarità attesa): {float(tit_val):.0f} "
                    f"< {cfg.CERTEZZA_TITOLARITA_SOGLIA:.0f}"
                )
                continue

        # Near-TOP diagnostics (FP high but blocked by VR / ML / experts)
        top_th = th.at[idx, "top_fp_mantra"]
        top_vr_th = th.at[idx, "top_vr"]
        fp_v = float(fp_mantra.at[idx])
        v = float(vr.at[idx])
        if fp_v > top_th:
            reasons = []
            if v <= top_vr_th:
                reasons.append(f"VR {v:.0f} <= soglia TOP VR {top_vr_th:.0f}")
            pred_val = None
            for col in ("predicted_next_fantavoto", "predicted_fantavoto"):
                if col in df.columns and pd.notna(df.at[idx, col]):
                    try:
                        pred_val = float(df.at[idx, col])
                    except (TypeError, ValueError):
                        pred_val = None
                    break
            if pred_val is not None:
                ruolo = df.at[idx, "ruolo_primario"] if "ruolo_primario" in df.columns else None
                soglia = cfg.NEXT_FANTAVOTO_MIN_BY_ROLE.get(ruolo, 6.0) if isinstance(ruolo, str) else 6.0
                if pred_val < soglia:
                    reasons.append(
                        f"predicted_next {pred_val:.2f} < soglia ruolo {soglia:.1f}"
                    )
            if "expert_rating" in df.columns and pd.notna(df.at[idx, "expert_rating"]):
                try:
                    er = float(df.at[idx, "expert_rating"])
                except (TypeError, ValueError):
                    er = None
                if er is not None and er < cfg.TOP_EXPERT_MIN:
                    reasons.append(
                        f"rating esperti {er:.0f} < {cfg.TOP_EXPERT_MIN:.0f}"
                    )
            if reasons:
                motivo.at[idx] = (
                    f"Quasi TOP (FP_Mantra {fp_v:.0f} > {top_th:.0f}) ma "
                    + "; ".join(reasons)
                )
                continue

        g = gap_rend.at[idx]
        if pd.notna(g):
            motivo.at[idx] = (
                f"Gap Rendimento (percentile VR-FP) {g:.0f}: serve "
                f"> {cfg.SCOMMESSA_GAP_MIN:.0f} per SCOMMESSA"
            )
        else:
            motivo.at[idx] = "Dati insufficienti per calcolare il gap Rendimento"
    return motivo


def _explain_prezzo(
    df: pd.DataFrame,
    gap_prezzo: pd.Series,
    label_prezzo: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Explain, for players with no Prezzo-axis label, why (in practice this
    is always "no quotation" — the three bands otherwise cover the whole
    gap range contiguously)."""
    motivo = pd.Series([None] * len(df), index=df.index, dtype=object)
    for idx in df.index[label_prezzo.isna()]:
        g = gap_prezzo.at[idx]
        if pd.isna(g):
            motivo.at[idx] = (
                "Nessuna quotazione disponibile: impossibile calcolare il gap "
                "prezzo/valore"
            )
        else:
            motivo.at[idx] = (
                f"Gap Prezzo/Valore (percentile prezzo-FP_Mantra) {g:.0f}: nessuna "
                f"fascia (serve <= -{cfg.GIUSTO_GAP_BAND:.0f} per AFFARE o "
                f">= {cfg.GIUSTO_GAP_BAND:.0f} per SOPRAVALUTATO)"
            )
    return motivo


def classify_fase7(
    df: pd.DataFrame,
    fp: pd.Series,
    fp_mantra: pd.Series,
    vr: pd.Series,
    p1: pd.Series,
    cfg: MantraConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Assign the two Fase7 axis labels, their reasons, and their raw gaps.

    Parameters
    ----------
    df:
        DataFrame with columns:
        - ``Stagioni_IT``, ``Pr``, ``DV`` — CERTEZZA (historical leg)
        - ``ruolo_primario`` — MANTRA primary role, used for all role pools
        - ``Pz1`` — current official quotation (``qt_a``); missing/<=0 means
          "never quoted", excluded from the Prezzo/Valore axis
        - ``titolarita_attesa`` (optional) — EWMA of matchday-status
          probability (0-100); CERTEZZA's forward-looking leg
        - ``predicted_next_fantavoto`` / ``predicted_fantavoto`` (optional)
          — ML projection; when present must meet role soglia for TOP
        - ``expert_rating`` (optional) — average expert score; when present
          must be >= ``cfg.TOP_EXPERT_MIN`` for TOP
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
    ``(label_rendimento, motivo_rendimento, gap_rendimento,
    label_prezzo, motivo_prezzo, gap_prezzo)`` — six ``pd.Series`` aligned to
    ``df.index``. The two gap series are always populated (when computable),
    even for labeled players, as a numeric confidence signal (e.g. for a
    frontend star rating) — see ``Fase7_*_Gap`` in ``ml/mantra/runner.py``.
    """
    th = _resolve_thresholds(df, fp_mantra, vr, cfg)
    pool_map = _build_pool_map(df, cfg)

    fp_pct = _pool_percentile(fp, df["ruolo_primario"], pool_map) * 100.0
    fp_mantra_pct = _pool_percentile(fp_mantra, df["ruolo_primario"], pool_map) * 100.0
    vr_pct = _pool_percentile(vr, df["ruolo_primario"], pool_map) * 100.0
    price = df.get("Pz1", pd.Series(np.nan, index=df.index))
    price_pct = _price_percentile(price, df["ruolo_primario"], pool_map)

    # ── Asse Rendimento/Affidabilità: TOP → CERTEZZA → SCOMMESSA ────────────
    label_rend = pd.Series([None] * len(df), index=df.index, dtype=object)

    ml_ok = _external_ml_ok(df, cfg)
    expert_ok = _external_expert_ok(df, cfg)
    mask = (
        (fp_mantra > th["top_fp_mantra"])
        & (vr > th["top_vr"])
        & ml_ok
        & expert_ok
    )
    label_rend[mask] = "TOP"

    stagioni = df.get("Stagioni_IT", pd.Series(0, index=df.index)).fillna(0)
    pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
    dv = df.get("DV", pd.Series(99, index=df.index)).fillna(99)
    certezza_storica = (
        (stagioni >= cfg.CERTEZZA_STAGIONI)
        & (pr >= cfg.CERTEZZA_PR)
        & (dv <= th["certezza_dv"])
        & (p1 >= cfg.CERTEZZA_P1)
    )
    tit = df.get("titolarita_attesa", pd.Series(np.nan, index=df.index))
    certezza_forward = (tit >= cfg.CERTEZZA_TITOLARITA_SOGLIA) & (dv <= th["certezza_dv"])
    mask = label_rend.isna() & (certezza_storica | certezza_forward)
    label_rend[mask] = "CERTEZZA"

    gap_rend = vr_pct - fp_pct
    mask = label_rend.isna() & (gap_rend > cfg.SCOMMESSA_GAP_MIN)
    label_rend[mask] = "SCOMMESSA"

    motivo_rend = _explain_rendimento(df, fp_mantra, vr, p1, gap_rend, th, label_rend, cfg)

    # ── Asse Prezzo/Valore: AFFARE / GIUSTO / SOPRAVALUTATO ─────────────────
    # Uses FP_Mantra (quality), not VR: VR already discounts expensive/strong
    # players via Fattore_Eroe, so gauging price against VR double-counts
    # cost and mislabels elite-and-pricey players as SOPRAVALUTATO even when
    # their price is fully earned (see module docstring).
    gap_prezzo = price_pct - fp_mantra_pct
    label_prezzo = pd.Series([None] * len(df), index=df.index, dtype=object)
    label_prezzo[gap_prezzo <= -cfg.GIUSTO_GAP_BAND] = "AFFARE"
    label_prezzo[gap_prezzo >= cfg.GIUSTO_GAP_BAND] = "SOPRAVALUTATO"
    mid_mask = gap_prezzo.notna() & label_prezzo.isna()
    label_prezzo[mid_mask] = "GIUSTO"

    motivo_prezzo = _explain_prezzo(df, gap_prezzo, label_prezzo, cfg)

    return label_rend, motivo_rend, gap_rend, label_prezzo, motivo_prezzo, gap_prezzo
