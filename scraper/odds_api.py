"""Scraper for Serie A winner odds via The Odds API (the-odds-api.com).

Replaces scraper/snai_odds.py as the source for the "Vincitore Serie A"
market: snai.it (and every other consumer bookmaker site tested — Eurobet,
Goldbet, Oddsportal) resets the TLS connection before completing the
handshake, consistent with datacenter-IP bot blocking that a purpose-built
API doesn't need to defend against.

The Odds API is a paid-tier-gated aggregator: it re-sells odds collected
from many bookmakers via one JSON endpoint, meant for programmatic access.
Free tier is 500 credits/month; each (sport, region, market) combination
costs 1 credit per call, so a single pre-season run here is cheap. Whether
"outrights" (futures) markets are available for soccer_italy_serie_a on the
free plan is NOT verified — the API may reject the request with an
"UNKNOWN_MARKET" or similar error if outrights aren't offered for this
sport/plan; this is surfaced as a clear log message rather than a crash.

NOTE: this module could not be tested against live data — outbound access
to the-odds-api.com was unreachable (TLS reset) from the environment this
was written in. The JSON parsing below is deliberately defensive (searches
for {name, price} outcome pairs rather than assuming an exact nesting) to
tolerate minor shape differences, but a first --dry-run against a real API
key is strongly recommended before wiring this into a scheduled job.

Usage:
    python -m scraper.odds_api --api-key YOUR_KEY --dry-run
    python -m scraper.odds_api --api-key YOUR_KEY --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_italy_serie_a"
SOURCE = "odds_api"

#: Canonical Serie A team names used throughout the rest of the pipeline
#: (same list as scraper/snai_odds.py's TEAM_MAP values).
CANONICAL_TEAMS: list[str] = [
    "Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta",
    "Fiorentina", "Bologna", "Torino", "Genoa", "Udinese", "Empoli", "Lecce",
    "Verona", "Cagliari", "Parma", "Como", "Sassuolo", "Pisa", "Cremonese",
    "Venezia", "Spezia", "Frosinone", "Salernitana", "Monza",
]

#: Prefixes/suffixes commonly used in official club names that the API may
#: return instead of the short form used internally (e.g. "AC Milan" -> "Milan").
_STRIP_PATTERNS = [
    r"^AC\s+", r"^AS\s+", r"^US\s+", r"^SSC\s+", r"^SS\s+", r"^Hellas\s+",
    r"^FC\s+", r"\s+Calcio.*$", r"\s+\d{4}$",
]


def _canonical_team_name(raw_name: str) -> Optional[str]:
    """Best-effort match of an API team name to our canonical short names."""
    name = raw_name.strip()
    for pattern in _STRIP_PATTERNS:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

    for canonical in CANONICAL_TEAMS:
        if name.lower() == canonical.lower():
            return canonical
    # Fallback: substring match (e.g. "Inter Milan" contains "Inter")
    for canonical in CANONICAL_TEAMS:
        if canonical.lower() in name.lower() or name.lower() in canonical.lower():
            return canonical
    return None


def _find_outcomes(node: Any, found: list[dict]) -> None:
    """Recursively collect {"name": ..., "price": ...} outcome dicts from an
    arbitrarily-nested JSON response, so we don't depend on knowing the exact
    nesting the API uses for outrights markets."""
    if isinstance(node, dict):
        if "name" in node and "price" in node:
            found.append(node)
        for v in node.values():
            _find_outcomes(v, found)
    elif isinstance(node, list):
        for item in node:
            _find_outcomes(item, found)


def scrape(
    api_key: str,
    season_start: Optional[int] = None,
    regions: str = "eu,uk",
) -> list[dict]:
    """Fetch Serie A winner outright odds and return per-team implied probabilities.

    Parameters
    ----------
    api_key:
        The Odds API key (https://the-odds-api.com).
    season_start:
        Season start year (defaults to current year).
    regions:
        Comma-separated bookmaker regions to aggregate across, for a more
        stable average than any single bookmaker.

    Returns
    -------
    List of dicts with keys: team, odds, implied_probability, season_start
    """
    if season_start is None:
        season_start = datetime.now().year

    url = f"{API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "outrights",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 422:
        log.error(
            "The Odds API rejected the request (422) — 'outrights' may not be "
            "available for %s on your plan. Response: %s", SPORT_KEY, resp.text[:500],
        )
        resp.raise_for_status()
    resp.raise_for_status()
    data = resp.json()

    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        log.info("The Odds API quota remaining: %s", remaining)

    outcomes: list[dict] = []
    _find_outcomes(data, outcomes)
    if not outcomes:
        log.warning("No outright outcomes found in API response for %s.", SPORT_KEY)
        return []

    # Average price per canonical team across all bookmakers/regions returned.
    prices_by_team: dict[str, list[float]] = {}
    unmatched: set[str] = set()
    for o in outcomes:
        try:
            price = float(o["price"])
        except (TypeError, ValueError):
            continue
        canonical = _canonical_team_name(str(o["name"]))
        if canonical is None:
            unmatched.add(str(o["name"]))
            continue
        prices_by_team.setdefault(canonical, []).append(price)

    if unmatched:
        log.warning("Could not map %d outcome name(s) to a known team: %s", len(unmatched), sorted(unmatched))

    records: list[dict] = []
    for team, prices in prices_by_team.items():
        avg_odds = sum(prices) / len(prices)
        implied_prob = round((1.0 / avg_odds) * 100, 2)
        records.append({
            "team": team,
            "odds": round(avg_odds, 2),
            "implied_probability": implied_prob,
            "season_start": season_start,
        })

    # Remove bookmaker overround so probabilities sum to ~100%.
    if records:
        total_prob = sum(r["implied_probability"] for r in records)
        if total_prob > 0:
            for r in records:
                r["implied_probability"] = round((r["implied_probability"] / total_prob) * 100, 2)

    log.info("Scraped odds for %d teams (season %d) from The Odds API.", len(records), season_start)
    return records


_UPSERT_SQL = """
    INSERT INTO team_season_odds (team, season_start, odds, implied_probability, source, scraped_at)
    VALUES (:team, :season_start, :odds, :implied_probability, :source, :scraped_at)
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
                    "source": SOURCE,
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

    parser = argparse.ArgumentParser(description="Scrape Serie A winner odds from The Odds API")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"), help="The Odds API key (or set ODDS_API_KEY)")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--season-start", type=int, default=None, help="Season start year")
    parser.add_argument("--regions", default="eu,uk", help="Comma-separated bookmaker regions to aggregate")
    parser.add_argument("--dry-run", action="store_true", help="Print records without persisting")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set the ODDS_API_KEY environment variable)")

    records = scrape(api_key=args.api_key, season_start=args.season_start, regions=args.regions)

    if args.dry_run:
        for r in records:
            log.info("  %s", r)
        log.info("Total: %d teams (dry-run)", len(records))
        return

    if args.db_url:
        n = persist(records, args.db_url)
        log.info("Persisted %d odds records to DB.", n)
    else:
        log.warning("No --db-url provided; use --dry-run to inspect data.")


if __name__ == "__main__":
    main()
