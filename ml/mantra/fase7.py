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
                 quantile ... OR an Indice Certezza Esperti (Gruppo Esperti
                 titolarita + salute, 1-10 each) >= CERTEZZA_ESPERTI_SOGLIA
                 (no DV gate on this leg — see below). Two independent
                 forward-looking legs for players who lack Serie A history
                 but have a locked-in starting spot, from two different
                 sources (scraped matchday odds vs. expert panel judgment).
3. 🔄 SCOMMESSA  percentile(VR) - percentile(FP), boosted by a Gruppo Esperti
                 quality index (bonus + media_voto, 1-10 each — never a
                 penalty, 0 when data is missing), > SCOMMESSA_GAP_MIN within
                 the role pool AND percentile(quotation) <=
                 SCOMMESSA_PREZZO_PERCENTILE_MAX (must actually be cheap — a
                 "scommessa" is a cheap gamble by definition; without the
                 price gate the label could land on an already-pricey
                 player, which defeats its purpose). Players with no
                 quotation never qualify (can't confirm "cheap").
4. (none)        see Fase7_Rendimento_Motivo for why.

The CERTEZZA-via-esperti leg skips the DV gate deliberately: its whole
purpose is covering players who lack the statistical history DV is built
from (DV defaults to a high "unreliable" value for them) — requiring DV
would defeat it for exactly the population it targets.

Axis 2: Prezzo/Valore (three contiguous bands of one gap)
-----------------------------------------------------------------------
Let ``gap = percentile(quotation) - percentile(quality)`` within the role
pool, where ``quality`` is FP_Mantra blended with the Gruppo Esperti TOTALE
score (``PREZZO_EXPERT_TOTALE_WEIGHT``, falls back to FP_Mantra alone when
TOTALE is missing for a player). Deliberately FP_Mantra, not VR: VR already
bakes in an anti-cost adjustment (``Fattore_Eroe`` discounts VR for
high-CP/expensive players, to reward cheap sleepers), so comparing
quotation against VR double-counts price — an elite AND expensive player
(high quotation percentile) gets a *dampened* VR percentile by design,
which would mislabel him SOPRAVALUTATO even though his price is fully
justified by his output. FP_Mantra carries no such cost adjustment, so the
gap here answers "is the price justified by how good he actually is", not
"is he a bargain relative to an already cost-adjusted score". The Gruppo
Esperti TOTALE blend exists to catch the opposite failure: a player whose
backward-looking FP_Mantra is temporarily depressed by bad luck (missed
penalties, hit the woodwork) but whom the expert panel still rates highly.
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


def _pool_percentile_of_valid(
    values: pd.Series,
    roles: pd.Series,
    pool_map: dict[str, set[str]],
) -> pd.Series:
    """Percentile (0-100) of each value within its role pool, excluding NaN
    from the reference population itself (not just from the result) — a
    player with no data for this metric doesn't get an arbitrary rank, and
    doesn't skew the ranks of those who do."""
    out = pd.Series(np.nan, index=values.index)
    for ruolo, pool_set in pool_map.items():
        mask = roles.isin(pool_set)
        pool_vals = values[mask].dropna()
        n = len(pool_vals)
        if n <= 1:
            continue
        ranks = pool_vals.rank(method="average") - 1.0
        out[pool_vals.index] = (ranks / (n - 1)) * 100.0
    return out


def _price_percentile(
    price: pd.Series,
    roles: pd.Series,
    pool_map: dict[str, set[str]],
) -> pd.Series:
    """Percentile (0-100) of each player's own quotation within his role
    pool, using only quoted players (price > 0) as the reference
    population. Never-quoted players get NaN — excluded from the pool
    population entirely, not given an arbitrary rank."""
    return _pool_percentile_of_valid(price.where(price > 0), roles, pool_map)


def _to_100(x: pd.Series) -> pd.Series:
    """Rescale a Gruppo Esperti 1-10 rating to 0-100 (1 -> 0, 10 -> 100)."""
    return (x - 1.0) / 9.0 * 100.0


def _expert_certezza_idx(df: pd.DataFrame, cfg: MantraConfig) -> pd.Series:
    """Indice Certezza Esperti (0-100): weighted blend of Gruppo Esperti
    ``titolarita`` + ``salute`` (both 1-10). NaN when either sub-score is
    missing for the player — the CERTEZZA-via-esperti leg simply doesn't
    fire for him, same "null = gate skipped" convention as the rest of this
    module."""
    if "expert_titolarita" not in df.columns or "expert_salute" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tit = pd.to_numeric(df["expert_titolarita"], errors="coerce")
    sal = pd.to_numeric(df["expert_salute"], errors="coerce")
    w = cfg.CERTEZZA_ESPERTI_TITOLARITA_PESO
    idx = w * _to_100(tit) + (1.0 - w) * _to_100(sal)
    return idx.where(tit.notna() & sal.notna())


def _expert_quality_idx(df: pd.DataFrame, cfg: MantraConfig) -> pd.Series:
    """Indice Qualità Esperti (0-100): weighted blend of Gruppo Esperti
    ``bonus_value`` + ``media_voto`` (both 1-10), feeding the SCOMMESSA gap
    boost. NaN when either sub-score is missing (no boost applied)."""
    if "expert_bonus" not in df.columns or "expert_media_voto" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    bonus = pd.to_numeric(df["expert_bonus"], errors="coerce")
    mv = pd.to_numeric(df["expert_media_voto"], errors="coerce")
    w = cfg.SCOMMESSA_QUALITY_BONUS_PESO
    idx = w * _to_100(bonus) + (1.0 - w) * _to_100(mv)
    return idx.where(bonus.notna() & mv.notna())


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
    gap_rend_eff: pd.Series,
    boost: pd.Series,
    price_pct: pd.Series,
    expert_certezza_idx: pd.Series,
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

        exp_idx_val = expert_certezza_idx.at[idx]
        if pd.notna(exp_idx_val):
            mancante_esperti = cfg.CERTEZZA_ESPERTI_SOGLIA - float(exp_idx_val)
            if 0 < mancante_esperti <= 10:
                motivo.at[idx] = (
                    f"Quasi CERTEZZA (via indice esperti): {float(exp_idx_val):.0f} "
                    f"< {cfg.CERTEZZA_ESPERTI_SOGLIA:.0f}"
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

        g = gap_rend_eff.at[idx]
        b = boost.at[idx]
        pp = price_pct.at[idx]
        if pd.notna(g):
            boost_note = f" (incluso boost esperti +{b:.0f})" if b > 0 else ""
            gap_ok = g > cfg.SCOMMESSA_GAP_MIN
            cheap_ok = pd.notna(pp) and pp <= cfg.SCOMMESSA_PREZZO_PERCENTILE_MAX
            if gap_ok and not cheap_ok:
                reason = (
                    f"percentile prezzo {pp:.0f} > {cfg.SCOMMESSA_PREZZO_PERCENTILE_MAX:.0f}"
                    if pd.notna(pp) else "nessuna quotazione"
                )
                motivo.at[idx] = (
                    f"Quasi SCOMMESSA (gap {g:.0f}{boost_note} > {cfg.SCOMMESSA_GAP_MIN:.0f}) "
                    f"ma non abbastanza economico: {reason}"
                )
            else:
                motivo.at[idx] = (
                    f"Gap Rendimento (percentile VR-FP{boost_note}) {g:.0f}: serve "
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
        - ``expert_titolarita`` / ``expert_salute`` (optional, 1-10 each) —
          Gruppo Esperti forward-looking CERTEZZA leg (no DV gate)
        - ``expert_bonus`` / ``expert_media_voto`` (optional, 1-10 each) —
          Gruppo Esperti quality boost on the SCOMMESSA gap (never negative)
        - ``expert_totale`` (optional, ~/50) — Gruppo Esperti blend into the
          Prezzo/Valore quality anchor alongside FP_Mantra
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
    ``gap_rendimento`` includes the Gruppo Esperti quality boost (never
    negative) so it stays consistent with whatever label it produced — a
    boost-assisted SCOMMESSA never shows a sub-threshold gap.
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
    expert_certezza_idx = _expert_certezza_idx(df, cfg)
    certezza_forward_esperti = expert_certezza_idx >= cfg.CERTEZZA_ESPERTI_SOGLIA  # niente gate DV
    mask = label_rend.isna() & (certezza_storica | certezza_forward | certezza_forward_esperti)
    label_rend[mask] = "CERTEZZA"

    gap_rend = vr_pct - fp_pct
    quality_idx = _expert_quality_idx(df, cfg)
    raw_boost = (
        (quality_idx - cfg.SCOMMESSA_QUALITY_NEUTRAL)
        / (100.0 - cfg.SCOMMESSA_QUALITY_NEUTRAL)
        * cfg.SCOMMESSA_QUALITY_BOOST_MAX
    )
    boost = raw_boost.clip(lower=0.0).fillna(0.0)  # mai negativo, mai un blocco
    gap_rend_eff = gap_rend + boost
    scommessa_cheap = price_pct.notna() & (price_pct <= cfg.SCOMMESSA_PREZZO_PERCENTILE_MAX)
    mask = label_rend.isna() & (gap_rend_eff > cfg.SCOMMESSA_GAP_MIN) & scommessa_cheap
    label_rend[mask] = "SCOMMESSA"

    motivo_rend = _explain_rendimento(
        df, fp_mantra, vr, p1, gap_rend_eff, boost, price_pct, expert_certezza_idx,
        th, label_rend, cfg,
    )

    # ── Asse Prezzo/Valore: AFFARE / GIUSTO / SOPRAVALUTATO ─────────────────
    # Uses FP_Mantra (quality), not VR: VR already discounts expensive/strong
    # players via Fattore_Eroe, so gauging price against VR double-counts
    # cost and mislabels elite-and-pricey players as SOPRAVALUTATO even when
    # their price is fully earned (see module docstring). Blended with the
    # Gruppo Esperti TOTALE percentile when available, so a player whose
    # FP_Mantra is temporarily depressed by bad luck isn't mislabeled either.
    expert_totale = df.get("expert_totale", pd.Series(np.nan, index=df.index))
    expert_totale_pct = _pool_percentile_of_valid(expert_totale, df["ruolo_primario"], pool_map)
    have_totale = expert_totale_pct.notna()
    w = cfg.PREZZO_EXPERT_TOTALE_WEIGHT
    quality_pct = fp_mantra_pct.where(
        ~have_totale, (1.0 - w) * fp_mantra_pct + w * expert_totale_pct
    )
    gap_prezzo = price_pct - quality_pct
    label_prezzo = pd.Series([None] * len(df), index=df.index, dtype=object)
    label_prezzo[gap_prezzo <= -cfg.GIUSTO_GAP_BAND] = "AFFARE"
    label_prezzo[gap_prezzo >= cfg.GIUSTO_GAP_BAND] = "SOPRAVALUTATO"
    mid_mask = gap_prezzo.notna() & label_prezzo.isna()
    label_prezzo[mid_mask] = "GIUSTO"

    motivo_prezzo = _explain_prezzo(df, gap_prezzo, label_prezzo, cfg)

    return label_rend, motivo_rend, gap_rend_eff, label_prezzo, motivo_prezzo, gap_prezzo
