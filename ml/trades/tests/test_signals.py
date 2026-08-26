"""Unit tests for forma / titolarità signal helpers."""

from __future__ import annotations

from ml.trades.signals import (
    MatchdayStatusRow,
    MatchdayVote,
    TitolaritaInputs,
    ewma_fantavoto,
    ewma_titolarita,
    forma_recente_score,
    indice_titolarita,
)


class TestEwma:
    def test_empty(self):
        ewma, n = ewma_fantavoto([])
        assert ewma is None and n == 0

    def test_single(self):
        ewma, n = ewma_fantavoto([MatchdayVote(1, 7.0)])
        assert n == 1
        assert abs(ewma - 7.0) < 1e-9

    def test_recency_bias(self):
        # Newer high vote should pull EWMA up vs older high vote
        recent_high = [
            MatchdayVote(5, 8.0),
            MatchdayVote(4, 6.0),
            MatchdayVote(3, 6.0),
        ]
        older_high = [
            MatchdayVote(5, 6.0),
            MatchdayVote(4, 6.0),
            MatchdayVote(3, 8.0),
        ]
        e1, _ = ewma_fantavoto(recent_high)
        e2, _ = ewma_fantavoto(older_high)
        assert e1 > e2

    def test_sv_skipped(self):
        votes = [
            MatchdayVote(3, None),
            MatchdayVote(2, 6.5),
            MatchdayVote(1, None),
        ]
        ewma, n = ewma_fantavoto(votes)
        assert n == 1
        assert abs(ewma - 6.5) < 1e-9


class TestFormaScore:
    def test_no_data(self):
        r = forma_recente_score([])
        assert r.forma is None
        assert r.confidence == "assente"
        assert r.games_available == 0

    def test_confidence_bands(self):
        one = [MatchdayVote(1, 6.5)]
        assert forma_recente_score(one).confidence == "bassa"
        three = [MatchdayVote(i, 6.5) for i in range(1, 4)]
        assert forma_recente_score(three).confidence == "media"
        five = [MatchdayVote(i, 6.5) for i in range(1, 6)]
        assert forma_recente_score(five).confidence == "alta"

    def test_scale_around_pool(self):
        # Exactly at pool mean → ~50
        votes = [MatchdayVote(i, 6.0) for i in range(1, 6)]
        r = forma_recente_score(votes, pool_mean=6.0, pool_std=0.8)
        assert r.forma is not None
        assert abs(r.forma - 50.0) < 1.0


class TestEwmaTitolarita:
    def test_empty(self):
        ewma, n = ewma_titolarita([])
        assert ewma is None and n == 0

    def test_all_starter_high_probability(self):
        rows = [MatchdayStatusRow(i, 90.0, "starter") for i in range(1, 4)]
        ewma, n = ewma_titolarita(rows)
        assert n == 3
        assert abs(ewma - 90.0) < 1e-6

    def test_injured_counts_as_zero_regardless_of_stored_probability(self):
        """An ongoing injury should pull the average down, not be ignored —
        even if the scraper still stored a stale non-zero probability."""
        rows = [
            MatchdayStatusRow(3, 85.0, "injured"),
            MatchdayStatusRow(2, 90.0, "starter"),
            MatchdayStatusRow(1, 90.0, "starter"),
        ]
        ewma, _ = ewma_titolarita(rows)
        # Most recent (highest weight) row is injured -> 0, pulling ewma well
        # below the all-starter case.
        all_starter, _ = ewma_titolarita(
            [MatchdayStatusRow(i, 90.0, "starter") for i in range(1, 4)]
        )
        assert ewma < all_starter

    def test_suspended_counts_as_zero(self):
        rows = [MatchdayStatusRow(1, 100.0, "suspended")]
        ewma, n = ewma_titolarita(rows)
        assert n == 1
        assert ewma == 0.0

    def test_recency_bias(self):
        recent_high = [
            MatchdayStatusRow(3, 90.0, "starter"),
            MatchdayStatusRow(2, 20.0, "bench"),
            MatchdayStatusRow(1, 20.0, "bench"),
        ]
        older_high = [
            MatchdayStatusRow(3, 20.0, "bench"),
            MatchdayStatusRow(2, 20.0, "bench"),
            MatchdayStatusRow(1, 90.0, "starter"),
        ]
        e1, _ = ewma_titolarita(recent_high)
        e2, _ = ewma_titolarita(older_high)
        assert e1 > e2

    def test_window_limits_lookback(self):
        rows = [MatchdayStatusRow(i, 90.0, "starter") for i in range(1, 10)]
        rows.append(MatchdayStatusRow(10, 0.0, "bench"))
        _, n = ewma_titolarita(rows, window=3)
        assert n == 3


class TestTitolarita:
    def test_both_signals(self):
        r = indice_titolarita(
            TitolaritaInputs(probability_matchday=80, titolarita_esperti=7)
        )
        # 0.6*80 + 0.4*70 = 48+28 = 76
        assert abs(r.indice - 76.0) < 0.5

    def test_prob_only(self):
        r = indice_titolarita(TitolaritaInputs(probability_matchday=90))
        assert abs(r.indice - 90.0) < 0.1

    def test_experts_only(self):
        r = indice_titolarita(TitolaritaInputs(titolarita_esperti=8))
        assert abs(r.indice - 80.0) < 0.1

    def test_missing_defaults_50(self):
        r = indice_titolarita(TitolaritaInputs())
        assert r.indice == 50.0

    def test_injury_flag(self):
        r = indice_titolarita(
            TitolaritaInputs(probability_matchday=0, status="injured")
        )
        assert any("Indisponibile" in f for f in r.flags)
