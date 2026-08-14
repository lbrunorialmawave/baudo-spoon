"""PR8 — regression tests for ``_candidate_players`` Gap B exclusion.

Bug recap (see docs/plan.md PR8): Gap B's "domestic player" exclusion
checked "has ANY Serie A row, ever" instead of "has a RECENT Serie A row
(target or target-1, i.e. what the runner's tier 1/2 already consumes)".
That conflated "never left Italy" (Berardi) with "used to play in Italy"
(Dragusin, Kolo Muani), silently starving the foreign scraper for anyone
who moved abroad after a Serie A spell.

These tests exercise the real ``_candidate_players`` SQL (not a
reimplementation of its logic) against an in-memory SQLite database whose
schema mirrors the columns the query actually touches:
``player_quotations``, ``player_id_map``, ``player_latest_stats_any_league``,
``player_stats_by_prediction_season``, ``player_season_stats``, ``seasons``,
``leagues``. The last four are real Postgres *views* in production
(migrations 018/025/026); here they're plain tables with the same column
shape, which is all ``_candidate_players`` depends on — it only ever reads
from them via ``EXISTS`` / column comparisons, never anything view-specific.

Table names deliberately match production so the SQL text under test is
copy-identical to what runs against Postgres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sa = pytest.importorskip("sqlalchemy")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.backfill_foreign_stats import _candidate_players  # noqa: E402

TARGET_SEASON = 2025  # e.g. listino 2025-26 (season_start = 2025)

_SCHEMA = """
CREATE TABLE player_quotations (
    fantacalcio_id INTEGER NOT NULL,
    season_start   INTEGER NOT NULL,
    player_name    TEXT NOT NULL
);

CREATE TABLE player_id_map (
    fantacalcio_id   INTEGER NOT NULL,
    season_start     INTEGER NOT NULL,
    player_fotmob_id INTEGER
);

CREATE TABLE player_latest_stats_any_league (
    fantacalcio_id INTEGER NOT NULL
);

CREATE TABLE player_stats_by_prediction_season (
    fantacalcio_id           INTEGER NOT NULL,
    prediction_season_start  INTEGER NOT NULL
);

CREATE TABLE leagues (
    id      INTEGER PRIMARY KEY,
    comp_id TEXT NOT NULL
);

CREATE TABLE seasons (
    id           INTEGER PRIMARY KEY,
    league_id    INTEGER NOT NULL,
    season_start INTEGER NOT NULL
);

CREATE TABLE player_season_stats (
    player_fotmob_id INTEGER NOT NULL,
    season_id        INTEGER NOT NULL
);
"""

SERIE_A_LEAGUE_ID = 1


@pytest.fixture()
def engine():
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(sa.text(stmt))
        conn.execute(
            sa.text("INSERT INTO leagues (id, comp_id) VALUES (:id, '55')"),
            {"id": SERIE_A_LEAGUE_ID},
        )
    return engine


def _add_quotation(conn, *, fantacalcio_id: int, season_start: int, name: str) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO player_quotations (fantacalcio_id, season_start, player_name) "
            "VALUES (:fid, :season, :name)"
        ),
        {"fid": fantacalcio_id, "season": season_start, "name": name},
    )


def _add_id_map(conn, *, fantacalcio_id: int, season_start: int, fotmob_id: int) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO player_id_map (fantacalcio_id, season_start, player_fotmob_id) "
            "VALUES (:fid, :season, :fotmob)"
        ),
        {"fid": fantacalcio_id, "season": season_start, "fotmob": fotmob_id},
    )


def _add_any_league_latest(conn, fotmob_id: int) -> None:
    conn.execute(
        sa.text("INSERT INTO player_latest_stats_any_league (fantacalcio_id) VALUES (:fid)"),
        {"fid": fotmob_id},
    )


def _add_prediction_row(conn, fotmob_id: int, prediction_season_start: int) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO player_stats_by_prediction_season "
            "(fantacalcio_id, prediction_season_start) VALUES (:fid, :season)"
        ),
        {"fid": fotmob_id, "season": prediction_season_start},
    )


def _add_serie_a_season(conn, fotmob_id: int, season_start: int) -> None:
    season_id = conn.execute(
        sa.text(
            "INSERT INTO seasons (league_id, season_start) VALUES (:lid, :season) "
            "RETURNING id"
        ),
        {"lid": SERIE_A_LEAGUE_ID, "season": season_start},
    ).scalar_one()
    conn.execute(
        sa.text(
            "INSERT INTO player_season_stats (player_fotmob_id, season_id) "
            "VALUES (:fid, :sid)"
        ),
        {"fid": fotmob_id, "sid": season_id},
    )


def _candidate_keys(engine) -> set[tuple[int, int]]:
    candidates = _candidate_players(engine, [TARGET_SEASON])
    return {(c.player_fotmob_id, c.target_season_start) for c in candidates}


def test_pure_domestic_player_is_not_a_candidate(engine):
    """Berardi-class: recent Serie A history, never left. Must stay excluded."""
    fotmob_id = 1001
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=1, season_start=TARGET_SEASON, name="Berardi")
        _add_id_map(conn, fantacalcio_id=1, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        _add_any_league_latest(conn, fotmob_id)
        _add_serie_a_season(conn, fotmob_id, TARGET_SEASON)  # recent Serie A row

    assert (fotmob_id, TARGET_SEASON) not in _candidate_keys(engine)


def test_ex_serie_a_now_abroad_is_a_candidate(engine):
    """Regression test for PR8 (Dragusin/Kolo Muani class).

    Serie A history is 2+ seasons old, player is now abroad with no
    foreign row persisted yet. Must be picked up so the career scraper
    actually runs for them — this is the scenario the pre-fix query
    silently dropped.
    """
    fotmob_id = 1002
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=2, season_start=TARGET_SEASON, name="Dragusin")
        _add_id_map(conn, fantacalcio_id=2, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        # Stale Serie A row: 2+ seasons before target, outside the
        # target / target-1 window already covered by runner tiers 1/2.
        _add_serie_a_season(conn, fotmob_id, TARGET_SEASON - 3)
        # This stale row is also what makes "player_latest_stats_any_league"
        # non-empty for him (his last known season, absent a foreign one).
        _add_any_league_latest(conn, fotmob_id)
        # No player_stats_by_prediction_season row for the target season
        # (foreign scraper has never run for him).

    assert (fotmob_id, TARGET_SEASON) in _candidate_keys(engine)


def test_never_seen_player_is_a_candidate_via_gap_a(engine):
    """Neo-arrivo never processed at all: Gap A, unaffected by this fix."""
    fotmob_id = 1003
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=3, season_start=TARGET_SEASON, name="Neo Arrivo")
        _add_id_map(conn, fantacalcio_id=3, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        # No any-league row, no Serie A row at all.

    assert (fotmob_id, TARGET_SEASON) in _candidate_keys(engine)


def test_ex_serie_a_just_returned_to_italy_is_not_a_candidate(engine):
    """Recent Serie A row (== target season): tier 1 already covers him."""
    fotmob_id = 1004
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=4, season_start=TARGET_SEASON, name="Rientrato")
        _add_id_map(conn, fantacalcio_id=4, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        _add_serie_a_season(conn, fotmob_id, TARGET_SEASON)  # current season Serie A
        _add_any_league_latest(conn, fotmob_id)

    assert (fotmob_id, TARGET_SEASON) not in _candidate_keys(engine)


def test_ex_serie_a_abroad_already_backfilled_is_not_a_candidate(engine):
    """Already has a foreign prediction row for this target: fully covered."""
    fotmob_id = 1005
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=5, season_start=TARGET_SEASON, name="Gia Backfillato")
        _add_id_map(conn, fantacalcio_id=5, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        _add_serie_a_season(conn, fotmob_id, TARGET_SEASON - 3)  # stale Serie A
        _add_any_league_latest(conn, fotmob_id)
        _add_prediction_row(conn, fotmob_id, TARGET_SEASON)  # foreign row already persisted

    assert (fotmob_id, TARGET_SEASON) not in _candidate_keys(engine)


def test_ex_serie_a_with_prior_season_recent_row_is_not_a_candidate(engine):
    """Serie A row at target-1 (not target itself): still within the tier 1/2
    window, so tier 2 (pss_prev) already covers him — must stay excluded."""
    fotmob_id = 1006
    with engine.begin() as conn:
        _add_quotation(conn, fantacalcio_id=6, season_start=TARGET_SEASON, name="Prev Season")
        _add_id_map(conn, fantacalcio_id=6, season_start=TARGET_SEASON, fotmob_id=fotmob_id)
        _add_serie_a_season(conn, fotmob_id, TARGET_SEASON - 1)
        _add_any_league_latest(conn, fotmob_id)

    assert (fotmob_id, TARGET_SEASON) not in _candidate_keys(engine)
