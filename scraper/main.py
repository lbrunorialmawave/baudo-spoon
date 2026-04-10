from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import settings
from src.db import Base, ingest_dataframe, ingest_league_stats, upsert_player_profiles
from src.logging_cfg import configure_logging
from src.models import LEAGUE_CATALOG, SERIE_A
from src.scraper import FotMobMatchStatsScraper
from src.stats_scraper import FotMobLeagueStatsScraper


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seasons_csv(value: str) -> list[str]:
    return _parse_csv_list(value)


def main() -> None:
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="FotMob match stats scraper")
    parser.add_argument(
        "--leagues",
        type=_parse_csv_list,
        default=[SERIE_A],
        metavar="LEAGUE,...",
        help="Comma-separated league names (e.g. 'Serie A,Premier League')",
    )
    parser.add_argument(
        "--seasons",
        type=_parse_seasons_csv,
        default=None,
        metavar="SEASON,...",
        help="Comma-separated seasons (e.g. '2023-2024,2024-2025'); omit for latest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_dir,
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database ingestion and write CSV files only (match stats mode)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help=(
            "Scrape per-season player & team ranking stats instead of match stats. "
            "Always requires a database connection."
        ),
    )
    parser.add_argument(
        "--roles",
        action="store_true",
        help=(
            "After scraping stats, fetch player role/position from FotMob playerData "
            "API and upsert into player_profiles table. Requires --stats."
        ),
    )
    args = parser.parse_args()

    if args.roles and not args.stats:
        parser.error("--roles requires --stats")

    if args.stats:
        _run_league_stats(args)
    else:
        _run_match_stats(args)


def _run_league_stats(args: argparse.Namespace) -> None:
    log = logging.getLogger(__name__)

    if not settings.database_url:
        raise SystemExit(
            "SCRAPER_DATABASE_URL is not set. "
            "Stats mode always requires a database connection."
        )

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    scraper = FotMobLeagueStatsScraper(
        leagues=args.leagues,
        seasons=args.seasons,
    )

    collected_player_ids: dict[int, str] = {}  # {player_fotmob_id: player_name}
    player_url_map: dict[int, str] = {}  # {player_fotmob_id: full_page_url}

    with Session(engine) as session:
        for (
            league_name,
            season_label,
            fotmob_season_id,
            stat_type,
            stat_category,
            rows,
        ) in scraper.run():
            meta = LEAGUE_CATALOG[league_name]
            count = ingest_league_stats(
                session=session,
                rows=rows,
                league_name=league_name,
                meta=meta,
                season_label=season_label,
                fotmob_season_id=fotmob_season_id,
                stat_type=stat_type,
                stat_category=stat_category,
            )
            log.info(
                "[%s] %s | %s / %s → %d rows",
                league_name,
                season_label,
                stat_type,
                stat_category,
                count,
            )
            if stat_type == "players":
                for r in rows:
                    collected_player_ids.setdefault(r["entity_id"], r["entity_name"])
                    if r.get("entity_url"):
                        player_url_map.setdefault(r["entity_id"], r["entity_url"])

    if getattr(args, "roles", False) and collected_player_ids:
        _fetch_and_upsert_roles(engine, collected_player_ids, player_url_map, log)


def _fetch_and_upsert_roles(
    engine: "sqlalchemy.Engine",
    player_ids: dict[int, str],
    player_url_map: dict[int, str],
    log: logging.Logger,
) -> None:
    """Fetch player roles via browser page navigation (bypasses x-mas token) and upsert."""
    from src.player_profile_scraper import fetch_player_profiles

    log.info(
        "role_fetch: fetching profiles for %d players (%d with known URLs)",
        len(player_ids), len(player_url_map),
    )
    profiles = fetch_player_profiles(player_ids, player_url_map)

    if profiles:
        from sqlalchemy.orm import Session

        with Session(engine) as session:
            upserted = upsert_player_profiles(session, profiles)
        log.info("role_fetch: upserted %d player profiles", upserted)


def _run_match_stats(args: argparse.Namespace) -> None:
    log = logging.getLogger(__name__)

    scraper = FotMobMatchStatsScraper(
        leagues=args.leagues,
        seasons=args.seasons,
        output_dir=args.output_dir,
    )

    if args.no_db:
        outputs = scraper.run()
        for (league_name, season_str), (_df, output_path) in outputs.items():
            log.info("[%s] season=%s  csv=%s", league_name, season_str, output_path)
        return

    if not settings.database_url:
        raise SystemExit(
            "SCRAPER_DATABASE_URL is not set. "
            "Either set the env var or run with --no-db to skip database ingestion."
        )

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        def on_round_complete(league_name: str, season_str: str, round_df: "pd.DataFrame") -> None:
            meta = LEAGUE_CATALOG[league_name]
            try:
                season_start = int(season_str.split("-")[0])
            except Exception:
                season_start = None
            count = ingest_dataframe(session, round_df, league_name, meta, season_start)
            log.info("[%s] season=%s  round rows ingested=%d", league_name, season_str, count)

        outputs = scraper.run(on_round_complete=on_round_complete)

    for (league_name, season_str), (_df, output_path) in outputs.items():
        log.info("[%s] season=%s  csv=%s", league_name, season_str, output_path)


if __name__ == "__main__":
    main()
