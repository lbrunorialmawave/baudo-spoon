#!/usr/bin/env python3
"""Historical backfill of foreign-player career snapshots (PR7 / plan §33–38).

Season-aware: each candidate carries (player, target_season_start) through to
the season resolver. Idempotent via unique constraints on leagues / seasons /
player_season_stats.

Rollout stages
--------------
  Stage 0  --baseline
  Stage 1  --shadow (classify + season-resolution metrics, no DB writes)
  Stage 2  persist one season:  --seasons 1
  Stage 3  persist two seasons: --seasons 2
  Stage 4  FOREIGN_PERSISTENCE_ENFORCE=1  (blocking threshold)
  Stage 5  --health  (post-backfill gates)

Usage examples
--------------
  # Stage 0 — baseline
  python -m scripts.backfill_foreign_stats --baseline

  # Stage 1 — shadow (no DB writes), last 2 seasons
  FOREIGN_SHADOW_MODE=1 python -m scripts.backfill_foreign_stats --seasons 2 --json

  # Stage 2 — one-season persist (warning-only coverage)
  python -m scripts.backfill_foreign_stats --seasons 1

  # Stage 3 — two-season persist
  python -m scripts.backfill_foreign_stats --seasons 2

  # Stage 4 — enforce
  FOREIGN_PERSISTENCE_ENFORCE=1 python -m scripts.backfill_foreign_stats --seasons 1

  # Stage 5 — health checks
  python -m scripts.backfill_foreign_stats --health

Safety checklist (plan §33)
---------------------------
  1. DB backup / recovery point
  2. --baseline counts before
  3. --shadow dry-run and review would_persist + season_* metrics
  4. Live backfill (1 season, then 2)
  5. --baseline after + --health
  6. Never blind DELETE on rollback
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
    """Row counts used for pre/post backfill comparison and health checks."""
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
        # Lineage coverage (migration 025) — -1 if column missing
        "pss_with_prediction_season": (
            "SELECT COUNT(*) FROM player_season_stats "
            "WHERE prediction_season_start IS NOT NULL"
        ),
        "pss_with_source_season": (
            "SELECT COUNT(*) FROM player_season_stats "
            "WHERE source_season_start IS NOT NULL"
        ),
        "pss_foreign_sentinel": (
            "SELECT COUNT(*) FROM player_season_stats "
            "WHERE fotmob_season_id = -1"
        ),
        # Target-aware view (migration 026) — -1 if view missing
        "player_stats_by_prediction_season": (
            "SELECT COUNT(*) FROM player_stats_by_prediction_season"
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


def _health_checks(engine: sa.Engine, result: Any | None = None) -> dict[str, Any]:
    """Post-backfill / on-demand health gates (plan §45).

    Returns a report with ``ok`` bool and per-gate status. Does not raise;
    caller decides exit code.
    """
    counts = _baseline_counts(engine)
    gates: list[dict[str, Any]] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        gates.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            log.error("health gate FAILED: %s — %s", name, detail)
        else:
            log.info("health gate ok: %s — %s", name, detail)

    # Schema / view presence
    gate(
        "latest_view_readable",
        counts.get("player_latest_stats_any_league", -1) >= 0,
        f"rows={counts.get('player_latest_stats_any_league')}",
    )
    gate(
        "target_aware_view_readable",
        counts.get("player_stats_by_prediction_season", -1) >= 0,
        f"rows={counts.get('player_stats_by_prediction_season')}",
    )

    # Lineage columns usable (NULL counts allowed for legacy rows)
    gate(
        "lineage_columns_queryable",
        counts.get("pss_with_prediction_season", -1) >= 0
        and counts.get("pss_with_source_season", -1) >= 0,
        f"prediction_set={counts.get('pss_with_prediction_season')} "
        f"source_set={counts.get('pss_with_source_season')}",
    )

    # Foreign sentinel rows are expected after any successful foreign persist
    # (soft gate: only fail if query itself failed)
    gate(
        "foreign_sentinel_queryable",
        counts.get("pss_foreign_sentinel", -1) >= 0,
        f"fotmob_season_id=-1 rows={counts.get('pss_foreign_sentinel')}",
    )

    if result is not None:
        # Conservation already asserted inside ForeignStatsResult; surface it.
        gate(
            "conservation_invariants",
            bool(getattr(result, "invariant_ok", False)),
            str(getattr(result, "invariant_errors", [])),
        )
        # Accounting: candidates accounted for
        c = result.candidates
        accounted = result.fetched + result.unresolved
        gate(
            "candidates_accounted",
            c == accounted,
            f"candidates={c} fetched+unresolved={accounted}",
        )
        # Season resolution accounting when we attempted any fetch
        if c > 0 and result.fetched == 0 and result.unresolved == c:
            # All unresolved is degraded-success, not a hard schema failure
            gate(
                "season_resolution_ran",
                True,
                "all candidates unresolved (degraded; check FotMob/network)",
            )
        elif result.fetched > 0:
            gate(
                "season_resolution_ran",
                (
                    result.season_target_selected
                    + result.season_previous_selected
                    + result.season_latest_selected
                    + result.season_no_valid
                )
                >= result.fetched,
                f"target={result.season_target_selected} "
                f"previous={result.season_previous_selected} "
                f"latest={result.season_latest_selected} "
                f"no_valid={result.season_no_valid}",
            )

    ok = all(g["ok"] for g in gates)
    return {"ok": ok, "gates": gates, "counts": counts}


def _candidate_players(
    engine: sa.Engine, season_starts: list[int]
) -> list:
    """Players in listino for the given seasons who lack any-league latest stats.

    Returns ``ForeignPlayerCandidate`` instances keyed by
    ``(player_fotmob_id, target_season_start)`` — the same player may appear
    once per target season. Never collapsed to ``dict[player_id] = name``.
    """
    from scraper.src.player_career_scraper import ForeignPlayerCandidate

    sql = sa.text("""
        SELECT DISTINCT
            pim.player_fotmob_id,
            pq.player_name,
            pq.season_start AS target_season_start
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

    seen: set[tuple[int, int]] = set()
    out: list[ForeignPlayerCandidate] = []
    for r in rows:
        pid = int(r["player_fotmob_id"])
        target = int(r["target_season_start"])
        key = (pid, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ForeignPlayerCandidate(
                player_fotmob_id=pid,
                player_name=r["player_name"],
                target_season_start=target,
                prediction_season_start=target,
            )
        )
    return out


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
        "--health",
        action="store_true",
        help="Run post-backfill health gates and exit (Stage 5 / plan §45)",
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

    if args.health:
        report = _health_checks(engine, result=None)
        if args.as_json:
            print(json.dumps({"health": report}, indent=2))
        else:
            print("=== health checks ===")
            print(f"  ok: {report['ok']}")
            for g in report["gates"]:
                status = "OK" if g["ok"] else "FAIL"
                print(f"  [{status}] {g['name']}: {g['detail']}")
        return 0 if report["ok"] else 1

    seasons = _resolve_seasons(engine, args.seasons)
    if not seasons:
        log.error("No seasons found in player_quotations")
        return 1

    log.info("Scanning seasons: %s", seasons)
    before = _baseline_counts(engine)
    candidates = _candidate_players(engine, seasons)
    log.info(
        "Candidates missing any-league stats: %d (unique player-season pairs)",
        len(candidates),
    )

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

    health = _health_checks(engine, result=result)

    report: dict[str, Any] = {
        "seasons": seasons,
        "shadow": shadow,
        "before": before,
        "after": after,
        "delta": {
            k: after[k] - before[k]
            for k in before
            if before[k] >= 0 and after.get(k, -1) >= 0
        },
        **result.to_dict(),
        # Explicit backfill metrics (plan §31 / §34)
        "foreign_players_discovered": result.candidates,
        "foreign_players_newly_persisted": result.persisted if not shadow else 0,
        "foreign_players_would_persist": result.would_persist if shadow else result.persisted,
        "foreign_players_unresolved": result.unresolved,
        "foreign_players_uncatalogued": result.uncatalogued,
        "health_ok": health["ok"],
        "health_gates": health["gates"],
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
        print(f"  target_selected:  {result.season_target_selected}")
        print(f"  previous_selected:{result.season_previous_selected}")
        print(f"  no_valid_season:  {result.season_no_valid}")
        print(f"  persistence_rate: {report['persistence_rate']}%")
        print(f"  invariant_ok:     {result.invariant_ok}")
        if not shadow:
            print("  delta:")
            for k, v in report["delta"].items():
                print(f"    {k}: {v:+d}")

    if not result.invariant_ok:
        log.error("Invariant failure: %s", result.invariant_errors)
        return 1

    if not health["ok"]:
        log.error("Health gates failed: %s", health["gates"])
        # Soft by default; hard when enforce is on
        enforce_health = os.environ.get("FOREIGN_PERSISTENCE_ENFORCE", "").lower() in (
            "1", "true", "yes",
        )
        if enforce_health:
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
