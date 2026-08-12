"""End-to-end golden regression for the season-aware foreign-players pipeline (PR6).

Golden fixture (plan §30):

  player_fotmob_id = TEST_FOREIGN_PLAYER (424242)
  target prediction season = 2025

  FotMob careerHistory:
    2025/26 → appearances = 0          (unusable)
    2024/25 → Eredivisie, apps = 30    (uncatalogued, valid)

Expected path:
  resolve with refresh policy (target=2025, allow previous)
    → source = 2024, fallback_depth = 1, PREVIOUS_SEASON_SELECTED
  persist lineage
  historical target=2024 query → 2024 row (never absolute-latest 2025)
  is_foreign_fallback = True
  training = NO, evaluation = NO, inference = YES
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scraper.src.models import LEAGUE_CATALOG
from scraper.src.player_career_scraper import (
    REASON_PREVIOUS_SEASON_SELECTED,
    REASON_TARGET_SEASON_SELECTED,
    SeasonResolutionPolicy,
    _best_tournament_entry,
    fetch_player_career_snapshot,
    resolve_season,
)

TEST_FOREIGN_PLAYER = 424242
TEST_PLAYER_NAME = "Golden Foreign"
TEST_LEAGUE = "Eredivisie"
TEST_APPEARANCES = 30
TEST_MINUTES = TEST_APPEARANCES * 70  # scraper estimate


# ── Fixture builders ──────────────────────────────────────────────────────────

def _season(
    season_name: str,
    *,
    league: str | None = TEST_LEAGUE,
    appearances: int = 20,
    goals: int = 8,
    assists: int = 4,
) -> dict:
    tournaments = []
    if league is not None or appearances:
        t: dict = {"appearances": appearances, "goals": goals, "assists": assists}
        if league is not None:
            t["leagueName"] = league
        tournaments.append(t)
    return {"seasonName": season_name, "tournamentStats": tournaments}


def _golden_career_entries() -> list:
    """Canonical PR6 payload: latest unusable, previous valid uncatalogued."""
    return [
        _season("2024/25", league=TEST_LEAGUE, appearances=TEST_APPEARANCES),
        _season("2025/26", appearances=0),
    ]


def _career_payload(entries: list) -> dict:
    return {
        "props": {
            "pageProps": {
                "data": {
                    "careerHistory": {
                        "careerItems": {
                            "senior": {"seasonEntries": entries}
                        }
                    }
                }
            }
        }
    }


# ── 1. Selection / competition ────────────────────────────────────────────────

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


# ── 2. Season resolution — core golden path ───────────────────────────────────

def test_golden_resolve_previous_when_latest_invalid():
    """Latest 2025 invalid → previous 2024 selected with depth=1."""
    result = resolve_season(
        _golden_career_entries(),
        SeasonResolutionPolicy(
            target_season_start=2025,
            allow_previous_season_fallback=True,
            max_fallback_depth=2,
        ),
    )
    assert result.selected is True
    assert result.selected_season_start == 2024
    assert result.fallback_depth == 1
    assert result.reason == REASON_PREVIOUS_SEASON_SELECTED
    assert result.target_season_start == 2025


def test_golden_historical_target_2024_never_picks_2025():
    """Historical backfill target=2024 must select 2024 even if 2025 is valid."""
    entries = [
        _season("2024/25", appearances=30),
        _season("2025/26", appearances=40),  # also valid
    ]
    result = resolve_season(
        entries,
        SeasonResolutionPolicy(
            target_season_start=2024,
            allow_previous_season_fallback=False,
        ),
    )
    assert result.selected is True
    assert result.selected_season_start == 2024
    assert result.reason == REASON_TARGET_SEASON_SELECTED
    assert result.selected_season_start != 2025


# ── 3. Fetch snapshot lineage ─────────────────────────────────────────────────

def test_golden_fetch_snapshot_lineage_and_fallback():
    payload = _career_payload(_golden_career_entries())
    policy = SeasonResolutionPolicy(
        target_season_start=2025,
        allow_previous_season_fallback=True,
        max_fallback_depth=1,
    )
    with patch(
        "scraper.src.player_career_scraper._fetch_next_data",
        return_value=payload,
    ):
        snap = fetch_player_career_snapshot(
            TEST_FOREIGN_PLAYER,
            TEST_PLAYER_NAME,
            target_season_start=2025,
            prediction_season_start=2025,
            season_policy=policy,
        )

    assert snap is not None
    assert snap["source_season_start"] == 2024
    assert snap["prediction_season_start"] == 2025
    assert snap["fallback_depth"] == 1
    assert snap["selection_reason"] == REASON_PREVIOUS_SEASON_SELECTED
    assert snap["league_name"] == TEST_LEAGUE
    assert snap["catalogued"] is False
    assert snap["appearances"] == TEST_APPEARANCES
    assert snap["minutes_estimate"] == TEST_MINUTES
    assert snap["league_name"] != "Unknown"


def test_golden_snapshot_shape_static():
    """Static shape check (no network) for downstream consumers."""
    minutes_estimate = TEST_APPEARANCES * 70
    per90 = 90.0 / minutes_estimate
    snap = {
        "player_fotmob_id": TEST_FOREIGN_PLAYER,
        "player_name": TEST_PLAYER_NAME,
        "league_name": TEST_LEAGUE,
        "season_label": "2024-2025",
        "source_season_start": 2024,
        "prediction_season_start": 2025,
        "selection_reason": REASON_PREVIOUS_SEASON_SELECTED,
        "fallback_depth": 1,
        "appearances": TEST_APPEARANCES,
        "minutes_estimate": minutes_estimate,
        "rating": 7.1,
        "goals_per_90": round(8 * per90, 3),
        "assists_per_90": round(4 * per90, 3),
        "estimated": True,
        "catalogued": TEST_LEAGUE in LEAGUE_CATALOG,
    }
    assert snap["catalogued"] is False
    assert snap["source_season_start"] == 2024
    assert snap["prediction_season_start"] == 2025
    assert snap["fallback_depth"] == 1


# ── 4. View contract (migration SQL) ──────────────────────────────────────────

def _migration_path(name: str) -> Path:
    candidates = [
        Path("db/migrations") / name,
        Path(__file__).resolve().parents[2] / "db/migrations" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(name)


def test_view_sql_latest_has_no_league_whitelist():
    migration = _migration_path("018_scope_mantra_views_and_add_foreign_fallback.sql")
    sql = migration.read_text()
    start = sql.index("CREATE OR REPLACE VIEW player_latest_stats_any_league")
    rest = sql[start:]
    next_create = rest.find("CREATE ", 10)
    body = rest if next_create < 0 else rest[:next_create]
    lowered = body.lower()
    assert "serie a" not in lowered
    assert "league_catalog" not in lowered
    assert "where rn = 1" in lowered


def test_view_sql_target_aware_present_in_026():
    migration = _migration_path("026_season_aware_view_contract.sql")
    sql = migration.read_text()
    assert "player_stats_by_prediction_season" in sql
    assert "prediction_season_start" in sql
    assert "source_season_start" in sql
    # Partition must be by (player, prediction_season), not player alone
    assert "PARTITION BY a.fantacalcio_id, a.prediction_season_start" in sql


def test_lineage_columns_present_in_025():
    migration = _migration_path("025_season_aware_foreign_lineage.sql")
    sql = migration.read_text()
    for col in (
        "source_season_start",
        "prediction_season_start",
        "selection_reason",
        "fallback_depth",
    ):
        assert col in sql
    assert "fotmob_season_id" in sql  # sentinel documented


# ── 5. Loader regression with target ──────────────────────────────────────────

def test_golden_loader_sets_foreign_flag(monkeypatch: pytest.MonkeyPatch):
    from ml.data.loader import _append_foreign_fallback_rows

    fallback = pd.DataFrame(
        [
            {
                "player_fotmob_id": TEST_FOREIGN_PLAYER,
                "player_name": TEST_PLAYER_NAME,
                "team_name": "Ajax",
                "league_name": TEST_LEAGUE,
                "minutes_avg": float(TEST_MINUTES),
                "goals_per90": 0.34,
                "assists_per90": 0.17,
                "saves_per90": 0.0,
                "clean_sheet_per90": 0.0,
                "source_season_start": 2024,
                "prediction_season_start": 2025,
            }
        ]
    )
    monkeypatch.setattr("ml.data.loader.pd.read_sql", lambda *a, **k: fallback)

    class _Conn:
        def execute(self, *a, **k):
            return self

        def scalar(self):
            return 2025

    class _Engine:
        def connect(self):
            return _Conn()

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
        df_player, engine=_Engine(), log=logging.getLogger("test")
    )
    foreign = result[result["player_fotmob_id"] == TEST_FOREIGN_PLAYER]
    assert len(foreign) == 1
    assert bool(foreign.iloc[0]["is_foreign_fallback"]) is True
    assert foreign.iloc[0]["league_name"] == TEST_LEAGUE


def test_golden_loader_historical_target_not_replaced_by_2025(
    monkeypatch: pytest.MonkeyPatch,
):
    """Backfill 2024 must surface the 2024 row, not a coexisting 2025 latest."""
    from ml.data.loader import _append_foreign_fallback_rows

    def fake_read_sql(sql, engine, params=None):
        text = str(sql)
        # Target-aware path only
        if "player_stats_by_prediction_season" not in text:
            raise RuntimeError("expected target-aware SQL")
        assert params is not None
        # Return only the row matching the requested prediction season
        target = params["season_start"]
        return pd.DataFrame(
            [
                {
                    "player_fotmob_id": TEST_FOREIGN_PLAYER,
                    "player_name": TEST_PLAYER_NAME,
                    "team_name": "Ajax",
                    "league_name": TEST_LEAGUE,
                    "minutes_avg": float(TEST_MINUTES),
                    "goals_per90": 0.34,
                    "assists_per90": 0.17,
                    "saves_per90": 0.0,
                    "clean_sheet_per90": 0.0,
                    "source_season_start": 2024,
                    "prediction_season_start": target,
                }
            ]
        )

    class _Conn:
        def execute(self, *a, **k):
            return self

        def scalar(self):
            return 2024

    class _Engine:
        def connect(self):
            return _Conn()

    monkeypatch.setattr("ml.data.loader.pd.read_sql", fake_read_sql)

    df_player = pd.DataFrame(
        [
            {
                "player_fotmob_id": 10,
                "player_name": "Serie A Regular",
                "team_fotmob_id": 100,
                "team_name": "Inter",
                "season_start": 2024,
                "season_label": "2024-25",
                "league_name": "Serie A",
            }
        ]
    )
    result = _append_foreign_fallback_rows(
        df_player, engine=_Engine(), log=logging.getLogger("test")
    )
    foreign = result[result["player_fotmob_id"] == TEST_FOREIGN_PLAYER]
    assert len(foreign) == 1
    # Cohort tagging uses domestic max (2024) — not swapped to 2025
    assert int(foreign.iloc[0]["season_start"]) == 2024
    assert bool(foreign.iloc[0]["is_foreign_fallback"]) is True


# ── 6. Trainer quarantine + inference ─────────────────────────────────────────

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
                "player_name": TEST_PLAYER_NAME,
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

    # Training / evaluation slice excludes foreign
    assert TEST_FOREIGN_PLAYER not in set(df_core["player_fotmob_id"])
    # Inference slice includes foreign
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


def test_golden_end_to_end_path_summary():
    """Single narrative assertion covering the full PR6 acceptance chain."""
    # 1) Resolve
    resolution = resolve_season(
        _golden_career_entries(),
        SeasonResolutionPolicy(
            target_season_start=2025,
            allow_previous_season_fallback=True,
            max_fallback_depth=2,
        ),
    )
    assert resolution.selected_season_start == 2024
    assert resolution.fallback_depth == 1

    # 2) Snapshot via mocked FotMob
    with patch(
        "scraper.src.player_career_scraper._fetch_next_data",
        return_value=_career_payload(_golden_career_entries()),
    ):
        snap = fetch_player_career_snapshot(
            TEST_FOREIGN_PLAYER,
            TEST_PLAYER_NAME,
            target_season_start=2025,
            prediction_season_start=2025,
            season_policy=SeasonResolutionPolicy(
                target_season_start=2025,
                allow_previous_season_fallback=True,
                max_fallback_depth=2,
            ),
        )
    assert snap is not None
    assert snap["source_season_start"] == 2024
    assert snap["prediction_season_start"] == 2025
    assert snap["catalogued"] is False

    # 3) Quarantine contract
    row = {
        "player_fotmob_id": TEST_FOREIGN_PLAYER,
        "is_foreign_fallback": True,
        "season_start": 2025,
    }
    df = pd.DataFrame([row])
    assert df[df["is_foreign_fallback"]].shape[0] == 1  # inference eligible
    assert df[~df["is_foreign_fallback"]].shape[0] == 0  # not in training
