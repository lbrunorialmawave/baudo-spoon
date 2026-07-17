"""Scraper for Snai betting odds — Serie A winner market.

Fetches https://www.snai.it/scommesse/quote/calcio/serie-a and extracts
the "Vincitore Serie A" (Serie A winner) odds for each team, converting
them to implied probabilities.

Usage:
    python -m scraper.snai_odds --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

URL = "https://www.snai.it/scommesse/quote/calcio/serie-a"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Teams to normalise (Snai names → canonical)
TEAM_MAP: dict[str, str] = {
    "Inter": "Inter",
    "Milan": "Milan",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Genoa": "Genoa",
    "Udinese": "Udinese",
    "Empoli": "Empoli",
    "Lecce": "Lecce",
    "Verona": "Verona",
    "Cagliari": "Cagliari",
    "Parma": "Parma",
    "Como": "Como",
    "Sassuolo": "Sassuolo",
    "Pisa": "Pisa",
    "Cremonese": "Cremonese",
    "Venezia": "Venezia",
    "Spezia": "Spezia",
    "Frosinone": "Frosinone",
    "Salernitana": "Salernitana",
    "Monza": "Monza",
}


def scrape(url: str = URL, season_start: Optional[int] = None) -> list[dict]:
    """Scrape Snai Serie A winner odds.

    Parameters
    ----------
    url:
        URL of the Snai Serie A odds page.
    season_start:
        Season start year (defaults to current year if before August,
        otherwise next year).

    Returns
    -------
    List of dicts with keys: team, odds, implied_probability, season_start
    """
    if season_start is None:
        now = datetime.now()
        season_start = now.year if now.month < 8 else now.year

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    html = resp.text
    records: list[dict] = []

    # Snai typically structures odds as JSON embedded in a script tag,
    # or as a table with team names and decimal odds.
    # Try to find the "Vincitore Serie A" market.

    # Pattern 1: look for decimal odds near team names
    # e.g. "Inter" followed by a decimal like "2.50" or fractional "1/2"
    for team_snai, team_canonical in TEAM_MAP.items():
        # Search for team name followed by odds
        pattern = re.compile(
            re.escape(team_snai) + r".*?(\d+\.\d+)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(html)
        if m:
            odds = float(m.group(1))
            # Implied probability = 1 / decimal odds
            implied_prob = round((1.0 / odds) * 100, 2)
        else:
            # Try fractional odds format: "Inter" then "5/1"
            frac_pattern = re.compile(
                re.escape(team_snai) + r".*?(\d+)/(\d+)",
                re.IGNORECASE | re.DOTALL,
            )
            fm = frac_pattern.search(html)
            if fm:
                numerator, denominator = float(fm.group(1)), float(fm.group(2))
                odds = (numerator / denominator) + 1 if denominator > 0 else 99.0
                implied_prob = round((1.0 / odds) * 100, 2)
            else:
                log.warning("Could not find odds for %s", team_snai)
                continue

        # Normalise: ensure probabilities sum reasonably (adjust for overround)
        records.append({
            "team": team_canonical,
            "odds": odds,
            "implied_probability": implied_prob,
            "season_start": season_start,
        })

    # Normalise probabilities (remove bookmaker overround / vigorish)
    if records:
        total_prob = sum(r["implied_probability"] for r in records)
        if total_prob > 0:
            for r in records:
                r["implied_probability"] = round(
                    (r["implied_probability"] / total_prob) * 100, 2
                )

    log.info(
        "Scraped Snai odds for %d teams (season %d).",
        len(records), season_start,
    )
    return records


_UPSERT_SQL = """
    INSERT INTO team_season_odds (team, season_start, odds, implied_probability, source, scraped_at)
    VALUES (:team, :season_start, :odds, :implied_probability, 'snai', :scraped_at)
    ON CONFLICT (team, season_start, source) DO UPDATE SET
        odds = EXCLUDED.odds,
        implied_probability = EXCLUDED.implied_probability,
        scraped_at = EXCLUDED.scraped_at
"""


def persist(records: list[dict], db_url: str) -> int:
    """Persist scraped odds to the database."""
    import sqlalchemy as sa

    engine = sa.create_engine(db_url)
    count = 0
    with engine.begin() as conn:
        for rec in records:
            conn.execute(
                sa.text(_UPSERT_SQL),
                {
                    "team": rec["team"],
                    "season_start": rec["season_start"],
                    "odds": rec["odds"],
                    "implied_probability": rec["implied_probability"],
                    "scraped_at": datetime.utcnow(),
                },
            )
            count += 1
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape Snai Serie A winner odds")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--url", default=URL, help="Snai odds page URL")
    parser.add_argument("--season-start", type=int, default=None, help="Season start year")
    parser.add_argument("--dry-run", action="store_true", help="Print records without persisting")
    args = parser.parse_args()

    records = scrape(url=args.url, season_start=args.season_start)

    if args.dry_run:
        for r in records:
            log.info("  %s", r)
        return

    if args.db_url:
        n = persist(records, args.db_url)
        log.info("Persisted %d odds records to DB.", n)
    else:
        log.warning("No --db-url provided; use --dry-run to inspect data.")


if __name__ == "__main__":
    main()
