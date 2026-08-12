"""PR4 reliability tests: seasonEntries selection, normalization, malformed payloads."""

from __future__ import annotations

import pytest

from scraper.src.db import normalize_league_name, _slugify_league
from scraper.src.models import LEAGUE_CATALOG, SERIE_A
from scraper.src.player_career_scraper import (
    ForeignStatsResult,
    _best_tournament_entry,
    _parse_season_start,
    _select_season_entry,
)


# ─── League name normalization ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Eredivisie", "Eredivisie"),
        ("  Eredivisie  ", "Eredivisie"),
        ("Eredivisie  ", "Eredivisie"),
        ("  Eredivisie", "Eredivisie"),
        ("Ere  divisie", "Ere divisie"),
        ("Serie A", "Serie A"),
    ],
)
def test_normalize_league_name(raw, expected):
    assert normalize_league_name(raw) == expected


def test_normalize_empty():
    assert normalize_league_name("   ") == ""
    assert normalize_league_name("") == ""


def test_slugify_league_deterministic():
    assert _slugify_league("Eredivisie") == "eredivisie"
    assert _slugify_league("Premier League") == "premier-league"
    assert _slugify_league("  La Liga  ") == "la-liga"
    assert _slugify_league("Eredivisie") == _slugify_league("Eredivisie")


# ─── seasonEntries selection (no more entries[0]) ─────────────────────────────


def test_select_season_most_recent():
    entries = [
        {"seasonName": "2022/2023", "appearances": 10},
        {"seasonName": "2024/2025", "appearances": 28},
        {"seasonName": "2023/2024", "appearances": 15},
    ]
    best = _select_season_entry(entries)
    assert best is not None
    assert best["seasonName"] == "2024/2025"


def test_select_season_unordered():
    entries = [
        {"seasonName": "2020/2021", "appearances": 5},
        {"seasonName": "2025/2026", "appearances": 1},
        {"seasonName": "2019/2020", "appearances": 30},
    ]
    best = _select_season_entry(entries)
    assert best["seasonName"] == "2025/2026"


def test_select_season_duplicate_keeps_max():
    entries = [
        {"seasonName": "2024/2025", "appearances": 10, "tag": "a"},
        {"seasonName": "2024/2025", "appearances": 20, "tag": "b"},
        {"seasonName": "2023/2024", "appearances": 30},
    ]
    best = _select_season_entry(entries)
    assert best["seasonName"] == "2024/2025"
    assert best["tag"] == "a"


def test_select_season_skips_malformed():
    entries = [
        {"seasonName": "unknown", "appearances": 99},
        {"seasonName": None, "appearances": 50},
        {"seasonName": "2023/2024", "appearances": 12},
        {"appearances": 40},
    ]
    best = _select_season_entry(entries)
    assert best is not None
    assert best["seasonName"] == "2023/2024"


def test_select_season_empty_or_all_bad():
    assert _select_season_entry([]) is None
    assert _select_season_entry([{"seasonName": "n/a"}]) is None
    assert _select_season_entry([None, "bad", 42]) is None  # type: ignore[list-item]


def test_parse_season_start_variants():
    assert _parse_season_start("2024/2025") == 2024
    assert _parse_season_start("2024-2025") == 2024
    assert _parse_season_start("2024") == 2024
    assert _parse_season_start("  2023/2024  ") == 2023
    assert _parse_season_start(None) is None
    assert _parse_season_start("") is None
    assert _parse_season_start("n/a") is None
    assert _parse_season_start(2024) is None  # type: ignore[arg-type]


# ─── Tournament selection robustness ─────────────────────────────────────────


def test_best_tournament_malformed_entries_safe():
    entry = {
        "tournamentStats": [
            "not-a-dict",
            {"leagueName": "Eredivisie", "appearances": "twelve"},
            {"leagueName": SERIE_A, "appearances": 8},
            None,
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == SERIE_A


def test_best_tournament_uncatalogued_selected():
    entry = {
        "tournamentStats": [
            {"leagueName": "Eredivisie", "appearances": 28},
        ]
    }
    best = _best_tournament_entry(entry)
    assert best["leagueName"] == "Eredivisie"
    assert "Eredivisie" not in LEAGUE_CATALOG


def test_best_tournament_empty():
    assert _best_tournament_entry({"tournamentStats": []}) is None
    assert _best_tournament_entry({}) is None


# ─── ForeignStatsResult invariants ───────────────────────────────────────────


def test_conservation_idempotent_shape():
    kwargs = dict(
        candidates=5, fetched=4, persisted=4, unresolved=1, uncatalogued=2
    )
    r1 = ForeignStatsResult(**kwargs)
    r2 = ForeignStatsResult(**kwargs)
    r1.assert_conservation()
    r2.assert_conservation()
    assert r1.invariant_ok is r2.invariant_ok is True
    assert r1.to_dict() == r2.to_dict()


def test_conservation_detects_drift():
    r = ForeignStatsResult(candidates=10, fetched=3, unresolved=3, persisted=3)
    r.assert_conservation()
    assert r.invariant_ok is False
    assert any("candidates" in e for e in r.invariant_errors)
