"""Unit tests for the Trade Fairness Engine."""

from __future__ import annotations

import pytest

from ml.trades.advisor import TradePlayer
from ml.trades.fairness import (
    DEFAULT_WEIGHTS,
    EnrichedTradePlayer,
    evaluate_trade,
    player_trade_value,
    validate_classic,
)


def _tp(
    pid: str,
    name: str,
    roles: set[str],
    fp: float = 60.0,
) -> TradePlayer:
    return TradePlayer(
        player_id=pid,
        name=name,
        eligible_roles=frozenset(roles),
        fp_corr=fp,
    )


def _enriched(
    tp: TradePlayer,
    *,
    forma: float | None = 55.0,
    games: int = 5,
    tit: float = 70.0,
    status: str = "starter",
    classic: str = "MID",
) -> EnrichedTradePlayer:
    return EnrichedTradePlayer(
        player=tp,
        forma_recente=forma,
        games_available_for_form=games,
        indice_titolarita=tit,
        status=status,
        classic_role=classic,
    )


class TestPlayerTradeValue:
    def test_identical_base_only(self):
        p = _enriched(_tp("1", "A", {"C"}, 60.0), forma=None, games=0, tit=60.0)
        r = player_trade_value(p)
        assert r.confidence == "assente"
        assert abs(r.score - 60.0) < 1.0

    def test_form_ramp(self):
        high = _enriched(_tp("1", "A", {"C"}, 50.0), forma=80.0, games=5, tit=50.0)
        low = _enriched(_tp("1", "A", {"C"}, 50.0), forma=80.0, games=1, tit=50.0)
        rh = player_trade_value(high)
        rl = player_trade_value(low)
        assert rh.score > rl.score
        assert rh.confidence == "alta"
        assert rl.confidence == "bassa"

    def test_injury_flag(self):
        p = _enriched(_tp("1", "A", {"C"}), status="injured")
        r = player_trade_value(p)
        assert any("Indisponibile" in f for f in r.flags)

    def test_bench_risk_flag(self):
        p = _enriched(_tp("1", "A", {"C"}), tit=30.0)
        r = player_trade_value(p)
        assert "Rischio panchina" in r.flags


class TestClassicValidation:
    def test_balanced_1v1(self):
        g = [_enriched(_tp("1", "A", {"C"}), classic="MID")]
        r = [_enriched(_tp("2", "B", {"C"}), classic="MID")]
        assert validate_classic(g, r) == []

    def test_cross_role_rejected(self):
        g = [_enriched(_tp("1", "A", {"A"}), classic="FWD")]
        r = [_enriched(_tp("2", "B", {"C"}), classic="MID")]
        errors = validate_classic(g, r)
        assert errors
        assert any("MID" in e or "FWD" in e for e in errors)

    def test_2_for_1_same_role_ok(self):
        g = [
            _enriched(_tp("1", "A", {"C"}), classic="MID"),
            _enriched(_tp("2", "B", {"C"}), classic="MID"),
        ]
        r = [_enriched(_tp("3", "C", {"C"}), classic="MID")]
        # count mismatch
        errors = validate_classic(g, r)
        assert any("pedine" in e.lower() or "MID" in e for e in errors)


class TestEvaluateTrade:
    def test_equal_trade_balanced(self):
        g = [_enriched(_tp("1", "A", {"C"}, 60.0), forma=60.0, tit=60.0, classic="MID")]
        r = [_enriched(_tp("2", "B", {"C"}, 60.0), forma=60.0, tit=60.0, classic="MID")]
        ev = evaluate_trade(mode="classic", give=g, receive=r, tolerance_percent=8.0)
        assert ev.valid
        assert ev.verdict == "equilibrato"
        assert abs(ev.value_delta_percent or 0) < 1.0

    def test_clear_upgrade(self):
        g = [_enriched(_tp("1", "A", {"C"}, 50.0), forma=50.0, tit=50.0, classic="MID")]
        r = [_enriched(_tp("2", "B", {"C"}, 80.0), forma=80.0, tit=80.0, classic="MID")]
        ev = evaluate_trade(mode="classic", give=g, receive=r, tolerance_percent=5.0)
        assert ev.valid
        assert ev.verdict == "vantaggioso"
        assert (ev.value_delta_percent or 0) > 5.0

    def test_invalid_classic_no_verdict(self):
        g = [_enriched(_tp("1", "A", {"A"}), classic="FWD")]
        r = [_enriched(_tp("2", "B", {"C"}), classic="MID")]
        ev = evaluate_trade(mode="classic", give=g, receive=r)
        assert not ev.valid
        assert ev.verdict is None
        assert ev.validation_errors
