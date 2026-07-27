"""One-shot backfill: copy existing ``match_method='manual'`` rows from
``player_id_map`` into the new ``manual_resolutions`` history table.

This script should be run once after applying migration 013 so that
pre-existing operator overrides are not lost — they become part of the
permanent history and are used as Pass 0 in future mapping pipelines.

Safe to re-run: uses ``ON CONFLICT DO NOTHING`` so duplicate associations
are silently skipped.

Usage::

    python scripts/backfill_manual_resolutions.py [--db-url POSTGRESQL_URL]

Environment variables (fallback order):
    ML_DATABASE_URL > API_DATABASE_URL > hardcoded default
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import sqlalchemy as sa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_manual_resolutions")

_INSERT_SQL = sa.text("""
    INSERT INTO manual_resolutions (
        fantacalcio_id, player_fotmob_id, season_start,
        name_fantacalcio, team_fantacalcio, canonical_role,
        name_fotmob, team_fotmob,
        note
    )
    VALUES (
        :fantacalcio_id, :player_fotmob_id, :season_start,
        :name_fantacalcio, :team_fantacalcio, :canonical_role,
        :name_fotmob, :team_fotmob,
        :note
    )
    ON CONFLICT (fantacalcio_id, player_fotmob_id) DO NOTHING
""")


def main() -> int:
    db_url = (
        os.environ.get("ML_DATABASE_URL")
        or os.environ.get("API_DATABASE_URL")
        or "postgresql://fbref:DevPassword123@db:5432/fbref"
    )
    engine = sa.create_engine(db_url, pool_pre_ping=True)

    # ── 1. Load manual rows from player_id_map ─────────────────────────────
    log.info("Loading manual rows from player_id_map …")
    manual = pd.read_sql(
        sa.text(
            "SELECT fantacalcio_id, season_start, player_fotmob_id, "
            "name_fantacalcio, team_fantacalcio, canonical_role, "
            "name_fotmob, team_fotmob "
            "FROM player_id_map "
            "WHERE match_method = 'manual' "
            "AND player_fotmob_id IS NOT NULL "
            "ORDER BY fantacalcio_id"
        ),
        engine,
    )
    log.info("  found %d manual rows", len(manual))

    if manual.empty:
        log.info("Nothing to backfill — no manual rows found.")
        return 0

    # ── 2. Prepare rows for insert ─────────────────────────────────────────
    manual = manual.astype(object).where(pd.notnull(manual), None)
    rows = []
    for _, row in manual.iterrows():
        rows.append({
            "fantacalcio_id": int(row["fantacalcio_id"]),
            "player_fotmob_id": int(row["player_fotmob_id"]),
            "season_start": int(row["season_start"]),
            "name_fantacalcio": str(row["name_fantacalcio"]),
            "team_fantacalcio": row.get("team_fantacalcio"),
            "canonical_role": row.get("canonical_role"),
            "name_fotmob": row.get("name_fotmob"),
            "team_fotmob": row.get("team_fotmob"),
            "note": "Backfilled from player_id_map (match_method=manual)",
        })

    # ── 3. Insert into manual_resolutions ──────────────────────────────────
    with engine.begin() as conn:
        conn.execute(_INSERT_SQL, rows)

    log.info("  inserted %d rows into manual_resolutions (skipped existing).", len(rows))
    log.info("Done! These resolutions are now preserved permanently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
