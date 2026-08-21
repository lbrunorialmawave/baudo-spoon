#!/usr/bin/env python3
"""One-shot backfill of player_matchday_votes from existing voti JSON files.

Example
-------
    export DATABASE_URL=postgresql://...
    PYTHONPATH=. python scripts/backfill_matchday_votes.py \\
        --voti-dir voti \\
        --seasons 2023-24,2024-25,2025-26
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("backfill_matchday_votes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voti-dir",
        type=Path,
        default=Path("voti"),
        help="Directory containing voti_fantacalcio-YYYY-YY.json",
    )
    parser.add_argument(
        "--seasons",
        default="2025-26",
        help="Comma-separated season labels to ingest",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: $DATABASE_URL or $ML_DATABASE_URL)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    db_url = (
        args.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("ML_DATABASE_URL")
    )
    if not db_url:
        log.error("DATABASE_URL / ML_DATABASE_URL not set")
        return 2

    # Reuse the loader CLI entrypoint
    from ml.data.voti_matchday_loader import main as loader_main

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    rc = 0
    for season in seasons:
        path = args.voti_dir / f"voti_fantacalcio-{season}.json"
        if not path.exists():
            log.error("Missing %s — skip", path)
            rc = 1
            continue
        log.info("=== Backfill %s from %s ===", season, path)
        code = loader_main(
            [
                "--json",
                str(path),
                "--season",
                season,
                "--database-url",
                db_url,
                *(["-v"] if args.verbose else []),
            ]
        )
        if code != 0:
            rc = code
            log.error("Loader failed for %s (exit %s)", season, code)
        else:
            log.info("OK %s", season)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
