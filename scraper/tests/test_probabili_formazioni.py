from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from scraper.probabili_formazioni import persist


def _init_db(db_url: str) -> None:
    engine = sa.create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE player_quotations (
                fantacalcio_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                season_start INTEGER NOT NULL
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE player_matchday_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fantacalcio_id INTEGER NOT NULL,
                season_start INTEGER NOT NULL,
                matchday INTEGER NOT NULL,
                team TEXT NOT NULL,
                probability INTEGER NOT NULL,
                status TEXT NOT NULL,
                injury_note TEXT,
                scraped_at TEXT NOT NULL,
                UNIQUE (fantacalcio_id, season_start, matchday)
            )
        """))
        conn.execute(
            sa.text(
                "INSERT INTO player_quotations (fantacalcio_id, player_name, team, season_start) "
                "VALUES (:fantacalcio_id, :player_name, :team, :season_start)"
            ),
            {
                "fantacalcio_id": 1234,
                "player_name": "Rovella",
                "team": "Lazio",
                "season_start": 2026,
            },
        )


def test_persist_resolves_fallback_and_skips_unmatched_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "probabili.sqlite"
    db_url = f"sqlite+pysqlite:///{db_path}"
    _init_db(db_url)

    records = [
        {
            "fantacalcio_id": None,
            "player_name": "Nicolò Rovella",
            "season_start": 2026,
            "matchday": 1,
            "team": "Lazio",
            "probability": 90,
            "status": "starter",
            "injury_note": None,
        },
        {
            "fantacalcio_id": None,
            "player_name": "Ghost Player",
            "season_start": 2026,
            "matchday": 1,
            "team": "Lazio",
            "probability": 10,
            "status": "bench",
            "injury_note": None,
        },
    ]

    persisted = persist(records, db_url)

    assert persisted == 1

    engine = sa.create_engine(db_url)
    with engine.begin() as conn:
        count = conn.scalar(sa.text("SELECT COUNT(*) FROM player_matchday_status"))
        row = conn.execute(
            sa.text(
                "SELECT fantacalcio_id, season_start, matchday, team, probability, status "
                "FROM player_matchday_status"
            )
        ).one()

    assert count == 1
    assert row.fantacalcio_id == 1234
    assert row.season_start == 2026
    assert row.matchday == 1
    assert row.team == "Lazio"
    assert row.probability == 90
    assert row.status == "starter"