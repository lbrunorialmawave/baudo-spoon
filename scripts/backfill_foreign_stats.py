#!/usr/bin/env python3
"""Historical backfill of foreign-player career snapshots (PR6 / plan §31–32).

Idempotent: unique constraints on leagues / seasons / player_season_stats mean
re-running is safe — already-present rows are upserted, not duplicated.

Usage examples:
  # Stage 1 — shadow (no DB writes), last 2 seasons
  FOREIGN_SHADOW_MODE=1 python -m scripts.backfill_foreign_stats --seasons 2

  # Stage 2 — persist with warning-only coverage
  python -m scripts.backfill_foreign_stats --seasons 2

  # Stage 4 — enforce persistence-rate threshold as hard failure
  FOREIGN_PERSISTENCE_ENFORCE=1 python -m scripts.backfill_foreign_stats --seasons 1

Safety checklist before production backfill (plan §32):
  1. Take a DB backup / recovery point
  2. Record baseline counts (printed by --baseline)
  3. Run with --shadow first and review would_persist
  4. Run without --shadow
  5. Compare post counts via --baseline again
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import sqlalchemy as sa

log = logging.getLogger("backfill_foreign_stats")


def _sync_db_url(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgres+asyncpg://", "postgres+psycopg2://")
        .replace("?ssl=", "?sslmode=")
        .replace("&ssl=", "&sslmode=")
    )


def _baseline_counts(engine: sa.Engine) -> dict[str, int]:
    """Row counts used for pre/post backfill comparison."""
    queries = {
        "leagues": "SELECT COUNT(*) FROM leagues",
        "seasons": "SELECT COUNT(*) FROM seasons",
        "player_season_stats": "SELECT COUNT(*) FROM player_season_stats",
        "player_latest_stats_any_league": (
            "SELECT COUNT(*) FROM player_latest_stats_any_league"
        ),
        # Uncatalogued leagues: those with NULL comp_id (post migration 024)
        "uncatalogued_leagues": (
            "SELECT COUNT(*) FROM leagues WHERE comp_id IS NULL"
        ),
    }
    out: dict[str, int] = {}
    with engine.connect() as conn:
        for key, sql in queries.items():
            try:
                out[key] = int(conn.execute(sa.text(sql)).scalar() or 0)
            except Exception as exc:  # noqa: BLE001
                log.warning("baseline %s failed: %s", key, exc)
                out[key] = -1
    return out


def _candidate_players(engine: sa.Engine, season_starts: list[int]) -> dict[int, str]:
    """Players in listino for the given seasons who lack any-league latest stats.

    Broader than season_refresh's "new this season only" filter: includes anyone
    in the quotation list for those seasons still missing a fallback row.
    """
    sql = sa.text("""
        SELECT DISTINCT pim.player_fotmob_id, pq.player_name
        FROM player_quotations pq
        JOIN player_id_map pim
          ON pim.fantacalcio_id = pq.fantacalcio_id
         AND pim.season_start = pq.season_start
        LEFT JOIN player_latest_stats_any_league pss_any
          ON pss_any.fantacalcio_id = pim.player_fotmob_id::bigint
        WHERE pq.season_start = ANY(:seasons)
          AND pim.player_fotmob_id IS NOT NULL
          AND pss_any.fantacalcio_id IS NULL
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"seasons": season_starts}).mappings().all()
    return {int(r["player_fotmob_id"]): r["player_name"] for r in rows}


def _resolve_seasons(engine: sa.Engine, n: int) -> list[int]:
    """Return the N most recent season_start values from player_quotations."""
    sql = sa.text("""
        SELECT DISTINCT season_start
        FROM player_quotations
        ORDER BY season_start DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"n": n}).scalars().all()
    return [int(s) for s in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill foreign-player career snapshots (idempotent)"
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("ML_DATABASE_URL"),
        help="Postgres URL (default: ML_DATABASE_URL)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=2,
        help="Number of most-recent quotation seasons to scan (default: 2)",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Classify only — no DB writes (Stage 1). Also set by FOREIGN_SHADOW_MODE=1",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print baseline row counts and exit (use before/after backfill)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit result as JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    if not args.db_url:
        log.error("Pass --db-url or set ML_DATABASE_URL")
        return 2

    engine = sa.create_engine(_sync_db_url(args.db_url), pool_pre_ping=True)

    if args.baseline:
        counts = _baseline_counts(engine)
        if args.as_json:
            print(json.dumps({"baseline": counts}, indent=2))
        else:
            print("=== baseline counts ===")
            for k, v in counts.items():
                print(f"  {k}: {v}")
        return 0

    seasons = _resolve_seasons(engine, args.seasons)
    if not seasons:
        log.error("No seasons found in player_quotations")
        return 1

    log.info("Scanning seasons: %s", seasons)
    before = _baseline_counts(engine)
    candidates = _candidate_players(engine, seasons)
    log.info("Candidates missing any-league stats: %d", len(candidates))

    shadow = args.shadow or os.environ.get("FOREIGN_SHADOW_MODE", "").lower() in (
        "1", "true", "yes",
    )

    from scraper.src.player_career_scraper import fetch_and_persist_players

    result = fetch_and_persist_players(
        candidates,
        _sync_db_url(args.db_url),
        shadow=shadow,
    )
    after = _baseline_counts(engine) if not shadow else before

    report: dict[str, Any] = {
        "seasons": seasons,
        "shadow": shadow,
        "before": before,
        "after": after,
        "delta": {k: after[k] - before[k] for k in before if before[k] >= 0 and after[k] >= 0},
        **result.to_dict(),
        # Explicit backfill metrics (plan §31)
        "foreign_players_discovered": result.candidates,
        "foreign_players_newly_persisted": result.persisted if not shadow else 0,
        "foreign_players_would_persist": result.would_persist if shadow else result.persisted,
        "foreign_players_unresolved": result.unresolved,
        "foreign_players_uncatalogued": result.uncatalogued,
    }

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print("=== backfill result ===")
        print(f"  seasons:          {seasons}")
        print(f"  shadow:           {shadow}")
        print(f"  candidates:       {report['foreign_players_discovered']}")
        print(f"  fetched:          {result.fetched}")
        if shadow:
            print(f"  would_persist:    {result.would_persist}")
            print(f"  would_skip:       {result.would_skip}")
        else:
            print(f"  newly_persisted:  {result.persisted}")
            print(f"  rows_written:     {result.rows_written}")
        print(f"  unresolved:       {result.unresolved}")
        print(f"  uncatalogued:     {result.uncatalogued}")
        print(f"  persistence_rate: {report['persistence_rate']}%")
        print(f"  invariant_ok:     {result.invariant_ok}")
        if not shadow:
            print("  delta:")
            for k, v in report["delta"].items():
                print(f"    {k}: {v:+d}")

    if not result.invariant_ok:
        log.error("Invariant failure: %s", result.invariant_errors)
        return 1

    enforce = os.environ.get("FOREIGN_PERSISTENCE_ENFORCE", "").lower() in (
        "1", "true", "yes",
    )
    rate = result.persistence_rate
    warn = float(os.environ.get("FOREIGN_PERSISTENCE_RATE_WARN", "0.90"))
    if enforce and rate is not None and rate < warn:
        log.error(
            "Persistence rate %.1f%% below enforce threshold %.0f%%",
            rate * 100, warn * 100,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
