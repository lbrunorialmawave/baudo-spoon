"""Tests for ml.mantra.scoring.compute_fp_corr, focused on the
Prezzo_Massimo formula anchoring on real per-role listino prices (Pz1)
instead of the role-blind CP composite."""

from __future__ import annotations

import pandas as pd

from ml.mantra.config import MantraConfig
from ml.mantra.scoring import compute_fp_corr


def _make_inputs(n_por=3, n_att=25, n_dif=25):
    """Build a synthetic pool: a small Por group plus large, well-separated
    Attaccanti/Difensori groups so both roles clear SOGLIA_POOL on their own
    and never fuse with each other."""
    roles = ["Por"] * n_por + ["A"] * n_att + ["Dc"] * n_dif
    n = len(roles)
    fp = pd.Series([50.0] * n)
    cp = pd.Series([50.0] * n)
    n_ruoli = pd.Series([1] * n)
    # Real listino prices: attackers priced far higher than defenders,
    # matching real auction economics.
    pz1 = pd.Series(
        [10.0] * n_por + [80.0] * n_att + [8.0] * n_dif
    )
    return roles, fp, cp, n_ruoli, pz1


def test_prezzo_massimo_reflects_role_price_gap() -> None:
    """Two players with identical VR (same fp/cp) in different roles must
    get different Prezzo_Massimo, proportional to their role's real price."""
    roles, fp, cp, n_ruoli, pz1 = _make_inputs()
    roles_s = pd.Series(roles)
    cfg = MantraConfig()

    result = compute_fp_corr(fp, cp, roles_s, n_ruoli, pz1, cfg)
    prezzo = result["prezzo_massimo"]

    att_prezzo = prezzo[roles_s == "A"].iloc[0]
    dif_prezzo = prezzo[roles_s == "Dc"].iloc[0]

    # Same FP/CP inputs -> same VR within each role's own pool (both
    # "average" for their role) -> price gap must come purely from the
    # real listino anchor (80 vs 8), not from CP.
    assert att_prezzo > dif_prezzo
    assert att_prezzo / dif_prezzo > 5


def test_neo_arrivo_with_zero_pz1_does_not_drag_role_average_down() -> None:
    roles, fp, cp, n_ruoli, pz1 = _make_inputs()
    roles_s = pd.Series(roles)
    cfg = MantraConfig()

    baseline = compute_fp_corr(fp, cp, roles_s, n_ruoli, pz1, cfg)["prezzo_massimo"]
    baseline_att = baseline[roles_s == "A"].iloc[0]

    # Add a neo-arrivo attacker with Pz1=0 (never quoted yet).
    roles2 = roles + ["A"]
    fp2 = pd.concat([fp, pd.Series([50.0])], ignore_index=True)
    cp2 = pd.concat([cp, pd.Series([50.0])], ignore_index=True)
    n_ruoli2 = pd.concat([n_ruoli, pd.Series([1])], ignore_index=True)
    pz12 = pd.concat([pz1, pd.Series([0.0])], ignore_index=True)
    roles2_s = pd.Series(roles2)

    with_neo = compute_fp_corr(fp2, cp2, roles2_s, n_ruoli2, pz12, cfg)["prezzo_massimo"]
    att_prices = with_neo[roles2_s == "A"]

    # Existing attackers' price should be unaffected by the neo-arrivo's 0.
    assert abs(att_prices.iloc[0] - baseline_att) < 1e-6
    # The neo-arrivo itself still gets a sensible (non-zero-anchored) price.
    assert att_prices.iloc[-1] > 0


def test_prezzo_massimo_has_a_floor_of_one() -> None:
    roles, fp, cp, n_ruoli, pz1 = _make_inputs()
    fp_low = pd.Series([0.0] * len(roles))
    cfg = MantraConfig()

    result = compute_fp_corr(fp_low, cp, pd.Series(roles), n_ruoli, pz1, cfg)
    assert (result["prezzo_massimo"] >= 1.0).all()
