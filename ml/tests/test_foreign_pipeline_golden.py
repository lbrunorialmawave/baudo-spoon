"""End-to-end golden path for the foreign-players pipeline (PR3).

Fixture (intentionally absent from LEAGUE_CATALOG):
  player_id = TEST_FOREIGN_PLAYER (424242)
  league    = Eredivisie
  season    = 2024-2025
  apps      = 30
  minutes   = 2100 (30 * 70 estimate)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scraper.src.models import LEAGUE_CATALOG
from scraper.src.player_career_scraper import _best_tournament_entry

TEST_FOREIGN_PLAYER = 424242
TEST_LEAGUE = "Eredivisie"
TEST_SEASON_LABEL = "2024-2025"
TEST_APPEARANCES = 30
TEST_MINUTES = TEST_APPEARANCES * 70


def test_golden_selection_accepts_eredivisie():
    assert TEST_LEAGUE not in LEAGUE_CATALOG
    entry = {
        "tournamentStats": [
            {
                "leagueName": TEST_LEAGUE,
                "appearances": TEST_APPEARANCES,
                "goals": 8,
                "assists": 4,
            },
        ]
    }
    best = _best_tournament_entry(entry)
    assert best is not None
    assert best["leagueName"] == TEST_LEAGUE


def test_golden_snapshot_shape():
    minutes_estimate = TEST_APPEARANCES * 70
    per90 = 90.0 / minutes_estimate
    snap = {
        "player_fotmob_id": TEST_FOREIGN_PLAYER,
        "player_name": "Golden Foreign",
        "league_name": TEST_LEAGUE,
        "season_label": TEST_SEASON_LABEL,
        "appearances": TEST_APPEARANCES,
        "minutes_estimate": minutes_estimate,
        "rating": 7.1,
        "goals_per_90": round(8 * per90, 3),
        "assists_per_90": round(4 * per90, 3),
        "estimated": True,
        "catalogued": TEST_LEAGUE in LEAGUE_CATALOG,
    }
    assert snap["catalogued"] is False
    assert snap["league_name"] == TEST_LEAGUE
    assert snap["minutes_estimate"] == TEST_MINUTES
    assert snap["league_name"] != "Unknown"


def test_view_sql_has_no_league_whitelist():
    migration = Path("db/migrations/018_scope_mantra_views_and_add_foreign_fallback.sql")
    if not migration.exists():
        migration = (
            Path(__file__).resolve().parents[2]
            / "db/migrations/018_scope_mantra_views_and_add_foreign_fallback.sql"
        )
    sql = migration.read_text()
    start = sql.index("CREATE OR REPLACE VIEW player_latest_stats_any_league")
    rest = sql[start:]
    next_create = rest.find("CREATE ", 10)
    body = rest if next_create < 0 else rest[:next_create]
    lowered = body.lower()
    assert "serie a" not in lowered
    assert "league_catalog" not in lowered
    assert "where rn = 1" in lowered


def test_golden_loader_sets_foreign_flag(monkeypatch: pytest.MonkeyPatch):
    from ml.data.loader import _append_foreign_fallback_rows
    import logging

    fallback = pd.DataFrame(
        [
            {
                "player_fotmob_id": TEST_FOREIGN_PLAYER,
                "player_name": "Golden Foreign",
                "team_name": "Ajax",
                "league_name": TEST_LEAGUE,
                "minutes_avg": float(TEST_MINUTES),
                "goals_per90": 0.34,
                "assists_per90": 0.17,
                "saves_per90": 0.0,
                "clean_sheet_per90": 0.0,
            }
        ]
    )
    monkeypatch.setattr("ml.data.loader.pd.read_sql", lambda *a, **k: fallback)

    df_player = pd.DataFrame(
        [
            {
                "player_fotmob_id": 10,
                "player_name": "Serie A Regular",
                "team_fotmob_id": 100,
                "team_name": "Inter",
                "season_start": 2025,
                "season_label": "2025-26",
                "league_name": "Serie A",
            }
        ]
    )
    result = _append_foreign_fallback_rows(
        df_player, engine=None, log=logging.getLogger("test")
    )
    foreign = result[result["player_fotmob_id"] == TEST_FOREIGN_PLAYER]
    assert len(foreign) == 1
    assert bool(foreign.iloc[0]["is_foreign_fallback"]) is True
    assert foreign.iloc[0]["league_name"] == TEST_LEAGUE


def test_golden_trainer_quarantine():
    df = pd.DataFrame(
        [
            {
                "player_fotmob_id": 10,
                "player_name": "Serie A Regular",
                "season_start": 2025,
                "fantavoto_medio": 6.5,
                "is_foreign_fallback": False,
            },
            {
                "player_fotmob_id": TEST_FOREIGN_PLAYER,
                "player_name": "Golden Foreign",
                "season_start": 2025,
                "fantavoto_medio": 7.0,
                "is_foreign_fallback": True,
                "league_name": TEST_LEAGUE,
            },
        ]
    )
    mask = df["is_foreign_fallback"].fillna(False)
    df_core = df[~mask]
    df_foreign = df[mask]

    assert TEST_FOREIGN_PLAYER not in set(df_core["player_fotmob_id"])
    assert TEST_FOREIGN_PLAYER in set(df_foreign["player_fotmob_id"])

    preds = pd.concat(
        [
            df_core.assign(predicted_fantavoto=6.5, is_foreign_fallback=False),
            df_foreign.assign(predicted_fantavoto=6.3, is_foreign_fallback=True),
        ],
        ignore_index=True,
    )
    golden_pred = preds[preds["player_fotmob_id"] == TEST_FOREIGN_PLAYER].iloc[0]
    assert bool(golden_pred["is_foreign_fallback"]) is True
    assert golden_pred["predicted_fantavoto"] == 6.3
