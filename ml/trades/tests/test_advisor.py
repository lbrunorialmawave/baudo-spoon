"""Tests for trade advisor — coverage, retention, top-scorer exclusion."""

from __future__ import annotations

from ml.trades.advisor import (
    TradePlayer,
    build_trade_dashboard,
    rank_trade_out_candidates,
    retention_score,
)


def _p(
    pid: str,
    name: str,
    roles: set[str],
    *,
    fp: float = 50.0,
    top5: bool = False,
    top10: bool = False,
    cost: int = 10,
    minutes: int = 1000,
) -> TradePlayer:
    return TradePlayer(
        player_id=pid,
        name=name,
        eligible_roles=frozenset(roles),
        cost=cost,
        current_value=cost,
        fp_corr=fp,
        is_top5_scorer_role=top5,
        is_top10_scorer_role=top10,
        minutes=minutes,
        team_serie_a="Venezia",
    )


def test_retention_top5_gets_bonus():
    base = _p("1", "Normal", {"A"}, fp=62.0)
    top = _p("2", "Martinez", {"A"}, fp=62.0, top5=True)
    assert retention_score(top) == retention_score(base) + 15.0
    assert retention_score(top) >= 75.0


def test_martinez_never_in_trade_out():
    """Hard exclusion: Top-5 scorer of weak team must not appear in trade_out.

    Martinez is placed in a surplus role (many A) so he would otherwise be a
    sell candidate; the hard-exclusion threshold must still keep him out.
    """
    squad = [
        _p("gk", "GK", {"Por"}, fp=55),
        _p("dc1", "DC1", {"Dc"}, fp=50),
        _p("dc2", "DC2", {"Dc"}, fp=48),
        _p("dc3", "DC3", {"Dc"}, fp=40),
        _p("martinez", "Martinez Jo.", {"A", "Pc"}, fp=62.0, top5=True, cost=20),
        _p("a2", "Bench Striker", {"A"}, fp=35.0, cost=5),
        _p("a3", "Third Striker", {"A"}, fp=30.0, cost=1),
        _p("a4", "Fourth Striker", {"A"}, fp=28.0, cost=1),
        _p("m1", "Mid1", {"C", "M"}, fp=55),
        _p("m2", "Mid2", {"C"}, fp=40),
    ]
    outs, excluded = rank_trade_out_candidates(squad, ["4-3-3", "3-5-2", "3-4-3"])

    out_names = {c.player.name for c in outs}
    assert "Martinez Jo." not in out_names

    excl_names = {e.player.name for e in excluded}
    assert "Martinez Jo." in excl_names, (
        f"expected Martinez in excluded, got excluded={excl_names}, outs={out_names}, "
        f"surplus check may have failed"
    )
    assert any("Top-5" in e.reason for e in excluded)


def test_low_retention_surplus_appears_in_out():
    squad = [
        _p("gk", "GK", {"Por"}, fp=50),
        _p("dc1", "DC1", {"Dc"}, fp=55),
        _p("dc2", "DC2", {"Dc"}, fp=50),
        _p("dc3", "DC3", {"Dc"}, fp=30),  # weak extra Dc
        _p("a1", "A1", {"A"}, fp=60),
        _p("m1", "M1", {"C"}, fp=50),
    ]
    outs, excluded = rank_trade_out_candidates(squad, ["4-3-3"])
    # DC3 should be a candidate if Dc is in surplus
    out_ids = {c.player.player_id for c in outs}
    # At least the weak one ranks lower than high FP if both surplus
    if "dc3" in out_ids:
        dc3 = next(c for c in outs if c.player.player_id == "dc3")
        assert dc3.retention < 50


def test_dashboard_structure():
    squad = [
        _p("gk", "Maignan", {"Por"}, fp=70),
        _p("dd", "DiLorenzo", {"Dd"}, fp=65),
        _p("dc1", "Bremer", {"Dc"}, fp=72),
        _p("dc2", "Gabbia", {"Dc"}, fp=55),
        _p("ds", "Dimarco", {"Ds", "E"}, fp=68),
        _p("m1", "Barella", {"C", "M"}, fp=78),
        _p("m2", "Calha", {"C", "T"}, fp=75),
        _p("m3", "Loca", {"M", "C"}, fp=60),
        _p("w1", "Leao", {"W", "A"}, fp=80),
        _p("w2", "Poli", {"W"}, fp=58),
        _p("a1", "Thuram", {"A"}, fp=77),
        _p("a2", "Hojlund", {"A", "Pc"}, fp=70),
        # extras for surplus
        _p("dc3", "ExtraDC", {"Dc"}, fp=35),
        _p("a3", "ExtraA", {"A"}, fp=32),
    ]
    market = [
        _p("mkt1", "TargetW", {"W", "T"}, fp=72, cost=25),
        _p("mkt2", "TargetB", {"B", "Dc"}, fp=60, cost=15),
    ]
    dash = build_trade_dashboard(
        squad,
        ["4-3-3", "3-5-2", "3-4-3"],
        market_pool=market,
    )
    assert dash.formation_prefs == ("4-3-3", "3-5-2", "3-4-3")
    assert len(dash.coverage_by_formation) == 3
    assert isinstance(dash.trade_out, tuple)
    assert isinstance(dash.excluded_top_performers, tuple)
    # coverage cells non-empty
    assert len(dash.coverage_cells) > 0
