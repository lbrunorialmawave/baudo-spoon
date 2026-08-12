"""PR4 idempotency & transaction-boundary unit tests."""

from __future__ import annotations

from itertools import permutations
from unittest.mock import MagicMock, patch

import pytest

from scraper.src.db import normalize_league_name
from scraper.src.player_career_scraper import (
    _best_tournament_entry,
    _persist_one_snapshot,
    _select_season_entry,
)


def test_normalization_collapses_whitespace_variants_to_same_key():
    variants = ["Eredivisie", " Eredivisie", "Eredivisie ", "  Eredivisie  "]
    keys = {normalize_league_name(v) for v in variants}
    assert keys == {"Eredivisie"}


def test_season_selection_stable_across_shuffles():
    entries = [
        {"seasonName": "2022/2023", "appearances": 10},
        {"seasonName": "2024/2025", "appearances": 28},
        {"seasonName": "2023/2024", "appearances": 15},
    ]
    for perm in permutations(entries):
        best = _select_season_entry(list(perm))
        assert best is not None
        assert best["seasonName"] == "2024/2025"


def test_tournament_selection_stable():
    entry = {
        "tournamentStats": [
            {"leagueName": "Eredivisie", "appearances": 20},
            {"leagueName": "Serie A", "appearances": 5},
        ]
    }
    a = _best_tournament_entry(entry)
    b = _best_tournament_entry(entry)
    assert a == b
    assert a["leagueName"] == "Serie A"


def test_persist_passes_commit_false_for_atomicity():
    """All stat-category upserts must share the session transaction."""
    snap = {
        "player_fotmob_id": 42,
        "player_name": "Test",
        "league_name": "Eredivisie",
        "season_label": "2024-2025",
        "minutes_estimate": 2100,
        "goals_per_90": 0.3,
        "assists_per_90": 0.1,
        "rating": 7.0,
    }
    session = MagicMock()
    import scraper.src.db as db_mod

    with patch.object(db_mod, "ingest_league_stats", return_value=1) as m:
        total = _persist_one_snapshot(session, snap)
        assert total == 4  # mins, goals, assists, rating
        assert m.call_count == 4
        for c in m.call_args_list:
            assert c.kwargs["commit"] is False


def test_persist_skips_blank_league_without_calling_ingest():
    snap = {
        "player_fotmob_id": 1,
        "player_name": "X",
        "league_name": "   ",
        "season_label": "2024-2025",
        "minutes_estimate": 100,
        "goals_per_90": 0,
        "assists_per_90": 0,
    }
    session = MagicMock()
    import scraper.src.db as db_mod

    with patch.object(db_mod, "ingest_league_stats") as m:
        total = _persist_one_snapshot(session, snap)
        assert total == 0
        m.assert_not_called()
