"""Unit tests for foreign-player career snapshot selection & observability (PR1+PR2)."""

from __future__ import annotations

from scraper.src.models import LEAGUE_CATALOG, SERIE_A
from scraper.src.player_career_scraper import (
    ForeignStatsResult,
    _best_tournament_entry,
)


def test_catalogued_league_only_selected():
    entry = {"tournamentStats": [{"leagueName": SERIE_A, "appearances": 20}]}
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == SERIE_A


def test_uncatalogued_league_only_selected():
    """Uncatalogued leagues must no longer be filtered out."""
    entry = {"tournamentStats": [{"leagueName": "Eredivisie", "appearances": 28}]}
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == "Eredivisie"
    assert "Eredivisie" not in LEAGUE_CATALOG


def test_catalogued_preferred_over_uncatalogued():
    entry = {
        "tournamentStats": [
            {"leagueName": "Eredivisie", "appearances": 30},
            {"leagueName": SERIE_A, "appearances": 10},
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == SERIE_A


def test_highest_appearances_among_catalogued():
    entry = {
        "tournamentStats": [
            {"leagueName": SERIE_A, "appearances": 5},
            {"leagueName": "Premier League", "appearances": 25},
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == "Premier League"


def test_empty_tournament_stats_returns_none():
    assert _best_tournament_entry({"tournamentStats": []}) is None
    assert _best_tournament_entry({}) is None


def test_malformed_tournament_ignored_safely():
    entry = {
        "tournamentStats": [
            {"leagueName": "Eredivisie"},  # no appearances key
            {"leagueName": SERIE_A, "appearances": 1},
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == SERIE_A


def test_entry_without_league_name_ranked_last():
    entry = {
        "tournamentStats": [
            {"appearances": 50},  # no leagueName
            {"leagueName": "Eredivisie", "appearances": 10},
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == "Eredivisie"


def test_foreign_stats_result_conservation_ok():
    r = ForeignStatsResult(
        candidates=10,
        fetched=8,
        persisted=7,
        unresolved=2,
        skipped_invalid=1,
    )
    r.assert_conservation()
    assert r.invariant_ok is True
    assert r.invariant_errors == []
    assert r.persistence_rate == 7 / 8


def test_foreign_stats_result_conservation_candidates_mismatch():
    r = ForeignStatsResult(
        candidates=10,
        fetched=5,
        unresolved=3,  # should be 5
        persisted=5,
    )
    r.assert_conservation()
    assert r.invariant_ok is False
    assert any("candidates" in e for e in r.invariant_errors)


def test_foreign_stats_result_conservation_fetched_mismatch():
    r = ForeignStatsResult(
        candidates=5,
        fetched=5,
        unresolved=0,
        persisted=3,
        skipped_invalid=0,
        skipped_other=0,  # 3 != 5
    )
    r.assert_conservation()
    assert r.invariant_ok is False
    assert any("fetched" in e or "accounted" in e for e in r.invariant_errors)


def test_foreign_stats_result_to_dict_shape():
    r = ForeignStatsResult(
        candidates=4,
        fetched=3,
        persisted=3,
        unresolved=1,
        uncatalogued=2,
    )
    r.assert_conservation()
    d = r.to_dict()
    assert d["candidates"] == 4
    assert d["fetched"] == 3
    assert d["persisted"] == 3
    assert d["unresolved"] == 1
    assert d["uncatalogued"] == 2
    assert d["persistence_rate"] == 100.0
    assert d["invariant_ok"] is True


def test_foreign_stats_result_zero_fetched_rate_is_none():
    r = ForeignStatsResult(candidates=5, fetched=0, unresolved=5)
    r.assert_conservation()
    assert r.persistence_rate is None
    assert r.to_dict()["persistence_rate"] is None
