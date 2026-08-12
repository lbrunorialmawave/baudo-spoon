from __future__ import annotations

"""Tests for ml.data.loader._append_foreign_fallback_rows — the cross-league
neo-arrivo fallback (players with zero Serie A history get one extra
inference-only row from their most recent season in any league, sourced from
player_latest_stats_any_league, migration 018)."""

import logging

import pandas as pd
import pytest

from ml.data.loader import _append_foreign_fallback_rows
from ml.preprocessing.features import add_per90_features


def _df_player(**overrides) -> pd.DataFrame:
    base = {
        "player_fotmob_id": [10, 20],
        "player_name": ["Existing A", "Existing B"],
        "team_fotmob_id": [100, 200],
        "team_name": ["Team A", "Team B"],
        "season_start": [2025, 2025],
        "season_label": ["2025-26", "2025-26"],
        "league_name": ["Serie A", "Serie A"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _fallback_row(player_fotmob_id: int = 99, **overrides) -> pd.DataFrame:
    base = {
        "player_fotmob_id": [player_fotmob_id],
        "player_name": ["Neo Arrivo"],
        "team_name": ["Team C"],
        "league_name": ["Premier League"],
        "minutes_avg": [1800.0],
        "goals_per90": [0.5],
        "assists_per90": [0.2],
        "saves_per90": [0.0],
        "clean_sheet_per90": [0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class _FakeEngine:
    """Placeholder — never actually touched, read_sql is monkeypatched."""


class _FakeConn:
    """Minimal context-manager-free stand-in for engine.connect().execute(...).scalar()."""

    def __init__(self, scalar_value):
        self._scalar_value = scalar_value

    def execute(self, *args, **kwargs):
        return self

    def scalar(self):
        return self._scalar_value


class _FakeEngineWithQuotations:
    """Engine whose .connect().execute(...).scalar() returns a fixed listino season."""

    def __init__(self, quotations_season: int):
        self._quotations_season = quotations_season

    def connect(self):
        return _FakeConn(self._quotations_season)


def test_excludes_players_already_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate already in df_player (e.g. matched by a wider query than
    intended) must never be duplicated."""
    monkeypatch.setattr(
        "ml.data.loader.pd.read_sql",
        lambda *a, **k: _fallback_row(player_fotmob_id=10),  # already in df_player
    )
    df_player = _df_player()
    result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))
    assert len(result) == len(df_player)
    assert not result["player_fotmob_id"].duplicated().any()


def test_injects_new_row_with_overridden_season_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ml.data.loader.pd.read_sql",
        lambda *a, **k: _fallback_row(),
    )
    df_player = _df_player()
    result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))
    assert len(result) == len(df_player) + 1

    new_row = result[result["player_fotmob_id"] == 99].iloc[0]
    # Overridden to the domestic pipeline's latest season, not whatever the
    # foreign view considers the player's own "most recent" season.
    assert new_row["season_start"] == df_player["season_start"].max()
    assert new_row["is_foreign_fallback"] is True


def test_existing_rows_flagged_non_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ml.data.loader.pd.read_sql",
        lambda *a, **k: _fallback_row(),
    )
    df_player = _df_player()
    df_player["is_foreign_fallback"] = False  # set by load_raw_data before calling
    result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))
    original_rows = result[result["player_fotmob_id"].isin([10, 20])]
    assert (original_rows["is_foreign_fallback"] == False).all()  # noqa: E712


def test_per90_roundtrip_through_add_per90_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw counts back-derived from the view's per-90 rates must, once
    run back through add_per90_features, reproduce the original per-90
    values — proving no changes are needed in features.py for these rows."""
    monkeypatch.setattr(
        "ml.data.loader.pd.read_sql",
        lambda *a, **k: _fallback_row(goals_per90=0.5, assists_per90=0.2),
    )
    df_player = _df_player()
    result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))

    recomputed = add_per90_features(result)
    new_row = recomputed[recomputed["player_fotmob_id"] == 99].iloc[0]
    assert new_row["goals_per90"] == pytest.approx(0.5)
    assert new_row["goal_assist_per90"] == pytest.approx(0.2)


def test_missing_view_degrades_to_noop(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _raise(*args, **kwargs):
        raise Exception("relation \"player_latest_stats_any_league\" does not exist")

    monkeypatch.setattr("ml.data.loader.pd.read_sql", _raise)
    df_player = _df_player()
    with caplog.at_level(logging.WARNING):
        result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))
    assert len(result) == len(df_player)
    assert any("migration 018" in r.message for r in caplog.records)


def test_empty_query_result_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.data.loader.pd.read_sql", lambda *a, **k: pd.DataFrame())
    df_player = _df_player()
    result = _append_foreign_fallback_rows(df_player, _FakeEngine(), logging.getLogger("test"))
    assert len(result) == len(df_player)


def test_uncatalogued_league_becomes_foreign_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3: uncatalogued league (Eredivisie) still surfaces as is_foreign_fallback=True."""
    monkeypatch.setattr(
        "ml.data.loader.pd.read_sql",
        lambda *a, **k: _fallback_row(
            player_fotmob_id=777,
            league_name="Eredivisie",
        ),
    )
    df_player = _df_player()
    result = _append_foreign_fallback_rows(
        df_player, _FakeEngine(), logging.getLogger("test")
    )
    assert len(result) == len(df_player) + 1
    foreign = result[result["player_fotmob_id"] == 777].iloc[0]
    assert foreign["is_foreign_fallback"] is True
    assert foreign["league_name"] == "Eredivisie"
    assert foreign["season_start"] == df_player["season_start"].max()


def test_listino_season_ahead_of_domestic_uses_listino_for_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: early in a new season, player_quotations/player_id_map
    are already on season N while player_season_stats for domestic Serie A is
    still on N-1 (matches not yet scraped). The SQL lookup must use the
    listino season (N) — that's what player_id_map/player_quotations are
    keyed on for a neo-arrivo transferred in during the transfer window —
    but the appended row must still be tagged with the domestic season (N-1)
    so it lands in the same "current squad" slice the trainer projects
    forward, alongside every other player.
    """
    captured_params: dict = {}

    def _fake_read_sql(query, engine, params=None, **kwargs):
        captured_params.update(params or {})
        return _fallback_row(player_fotmob_id=823825, league_name="Premier League")

    monkeypatch.setattr("ml.data.loader.pd.read_sql", _fake_read_sql)

    df_player = _df_player()  # domestic season_start = 2025
    fake_engine = _FakeEngineWithQuotations(quotations_season=2026)

    result = _append_foreign_fallback_rows(
        df_player, fake_engine, logging.getLogger("test")
    )

    # The SQL param sent to the DB must be the listino season (2026), or the
    # neo-arrivo (mapped only under season_start=2026 in player_id_map /
    # player_quotations) would never be found.
    assert captured_params.get("season_start") == 2026

    # But the row appended to the dataframe must stay on the domestic
    # season (2025) so it's part of the same cohort as the other players
    # when the trainer selects df[df.season_start == df.season_start.max()].
    new_row = result[result["player_fotmob_id"] == 823825].iloc[0]
    assert new_row["season_start"] == 2025
    assert new_row["is_foreign_fallback"] is True

    # Domestic players must remain in the same latest-season slice as the
    # neo-arrivo — this is the exact regression this test guards against.
    latest = result["season_start"].max()
    assert latest == 2025
    assert set(result[result["season_start"] == latest]["player_fotmob_id"]) == {
        10, 20, 823825,
    }