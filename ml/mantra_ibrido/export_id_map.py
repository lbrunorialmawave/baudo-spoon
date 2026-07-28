"""Export the player_id_map table to a JSON file for the hybrid merger.

Usage
-----
python -m ml.mantra_ibrido.export_id_map  \
    --db-url "postgresql+psycopg2://user:pass@host:port/db?sslmode=require" \
    --output artifacts/player_id_map.json

The output is a JSON array of ``{"fantacalcio_id": …, "player_fotmob_id": …}``
objects, one per row where ``player_fotmob_id IS NOT NULL``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sqlalchemy as sa

log = logging.getLogger(__name__)

_SQL = sa.text("""
    SELECT fantacalcio_id, player_fotmob_id, season_start, match_method
    FROM player_id_map
    WHERE player_fotmob_id IS NOT NULL
    ORDER BY season_start DESC, fantacalcio_id
""")


def export_id_map(db_url: str, output: Path) -> None:
    """Query ``player_id_map`` and write a JSON array to *output*."""
    engine = sa.create_engine(db_url)
    with engine.begin() as conn:
        rows = conn.execute(_SQL).fetchall()

    records = [
        {
            "fantacalcio_id": int(r.fantacalcio_id),
            "player_fotmob_id": int(r.player_fotmob_id),
            "season_start": int(r.season_start),
            "match_method": r.match_method,
        }
        for r in rows
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    log.info("Exported %d ID-map entries to %s", len(records), output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export player_id_map to JSON")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/player_id_map.json"),
        help="Output JSON path (default: artifacts/player_id_map.json)",
    )
    args = parser.parse_args()

    export_id_map(args.db_url, args.output)
