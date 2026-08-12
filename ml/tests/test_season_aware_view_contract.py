"""PR5 — season-aware view / historical target contract.

These tests lock the consumer-side contract without requiring a live DB:
  - target-aware SQL filters on prediction_season_start
  - latest-absolute SQL remains available as fallback
  - coexisting 2024 + 2025 rows must not collapse to absolute latest when
    the target is historical
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.data.loader import (
    _FOREIGN_FALLBACK_SQL_LATEST,
    _FOREIGN_FALLBACK_SQL_TARGET_AWARE,
    _append_foreign_fallback_rows,
)


def test_target_aware_sql_filters_prediction_season():
    sql = _FOREIGN_FALLBACK_SQL_TARGET_AWARE
    assert "player_stats_by_prediction_season" in sql
    assert "prediction_season_start = :season_start" in sql
    assert "source_season_start" in sql


def test_latest_sql_still_uses_absolute_latest_view():
    sql = _FOREIGN_FALLBACK_SQL_LATEST
    assert "player_latest_stats_any_league" in sql
    # Must NOT hard-filter prediction_season (legacy absolute path)
    assert "prediction_season_start = :season_start" not in sql


def test_append_prefers_target_aware_then_falls_back(monkeypatch: pytest.MonkeyPatch):
    """First read_sql uses target-aware; on failure, latest is attempted."""
    calls: list[str] = []

    def fake_read_sql(sql, engine, params=None):
        text = str(sql)
        if "player_stats_by_prediction_season" in text:
            calls.append("target")
            raise RuntimeError("view missing")
        if "player_latest_stats_any_league" in text:
            calls.append("latest")
            return pd.DataFrame(
                {
                    "player_fotmob_id": [99],
                    "player_name": ["Neo"],
                    "team_name": ["X"],
                    "league_name": ["Eredivisie"],
                    "minutes_avg": [2000.0],
                    "goals_per90": [0.4],
                    "assists_per90": [0.1],
                    "saves_per90": [0.0],
                    "clean_sheet_per90": [0.0],
                }
            )
        raise AssertionError(f"unexpected SQL: {text[:80]}")

    class _Conn:
        def execute(self, *a, **k):
            return self

        def scalar(self):
            return 2025

    class _Engine:
        def connect(self):
            return _Conn()

    monkeypatch.setattr("ml.data.loader.pd.read_sql", fake_read_sql)

    df_player = pd.DataFrame(
        {
            "player_fotmob_id": [10],
            "player_name": ["Domestic"],
            "team_fotmob_id": [1],
            "team_name": ["A"],
            "season_start": [2025],
            "season_label": ["2025-26"],
            "league_name": ["Serie A"],
        }
    )
    out = _append_foreign_fallback_rows(df_player, _Engine(), __import__("logging").getLogger("t"))
    assert calls == ["target", "latest"]
    assert (out["is_foreign_fallback"] == True).sum() == 1
    assert 99 in set(out["player_fotmob_id"])


def test_coexisting_seasons_target_aware_selects_matching_row(
    monkeypatch: pytest.MonkeyPatch,
):
    """When target-aware view returns the 2024 row, loader must not swap in 2025."""

    def fake_read_sql(sql, engine, params=None):
        text = str(sql)
        assert "player_stats_by_prediction_season" in text
        # Simulate DB already filtered by prediction_season_start = :season_start
        assert params is not None and params["season_start"] == 2024
        return pd.DataFrame(
            {
                "player_fotmob_id": [823825],
                "player_name": ["Kolo Muani"],
                "team_name": ["PSG"],
                "league_name": ["Ligue 1"],
                "minutes_avg": [2500.0],
                "goals_per90": [0.35],
                "assists_per90": [0.15],
                "saves_per90": [0.0],
                "clean_sheet_per90": [0.0],
                "source_season_start": [2024],
                "prediction_season_start": [2024],
            }
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

    # Domestic frame empty for this player; season max forced via a decoy row
    # that will be filtered only by id presence — use a different player.
    df_player = pd.DataFrame(
        {
            "player_fotmob_id": [1],
            "player_name": ["Other"],
            "team_fotmob_id": [1],
            "team_name": ["A"],
            "season_start": [2024],
            "season_label": ["2024-25"],
            "league_name": ["Serie A"],
        }
    )
    out = _append_foreign_fallback_rows(df_player, _Engine(), __import__("logging").getLogger("t"))
    foreign = out[out["player_fotmob_id"] == 823825]
    assert len(foreign) == 1
    assert bool(foreign.iloc[0]["is_foreign_fallback"]) is True
    # Tagged with domestic/output cohort season (contract unchanged)
    assert int(foreign.iloc[0]["season_start"]) == 2024
