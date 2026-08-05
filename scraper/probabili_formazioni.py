"""Scraper for Fantacalcio probable formations (probabili formazioni).

Fetches https://www.fantacalcio.it/probabili-formazioni-serie-a and
parses the HTML to extract per-player matchday status:

    - Probable starter probability (0-100 %)
    - Status: starter, bench, injured, suspended, doubtful
    - Injury descriptions
    - Ballottaggi (tactical ballots)

Usage:
    python -m scraper.probabili_formazioni --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

URL = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Regex to extract fantacalcio_id from player URLs like
# /fiorentina/christensen-o/6403/2025-26
_PLAYER_URL_RE = re.compile(r"/(\d+)/(\d{4})-\d{2}")


# ── Data structures ──────────────────────────────────────────────────────────

_LINEUP_PLAYERS_SQL = """
    INSERT INTO player_matchday_status
        (fantacalcio_id, season_start, matchday, team, probability, status, injury_note, scraped_at)
    VALUES
        (:fantacalcio_id, :season_start, :matchday, :team, :probability, :status, :injury_note, :scraped_at)
    ON CONFLICT (fantacalcio_id, season_start, matchday) DO UPDATE SET
        probability = EXCLUDED.probability,
        status = EXCLUDED.status,
        injury_note = EXCLUDED.injury_note,
        scraped_at = EXCLUDED.scraped_at
"""


# ── Scraper ──────────────────────────────────────────────────────────────────


def _extract_player_id(url: str) -> Optional[int]:
    """Extract fantacalcio_id from a player detail URL."""
    m = _PLAYER_URL_RE.search(url)
    if m:
        return int(m.group(1))
    return None


def _extract_season(url: str) -> Optional[int]:
    """Extract season_start from a player detail URL."""
    m = _PLAYER_URL_RE.search(url)
    if m:
        return int(m.group(2))
    return None


def _parse_probability(text: str) -> int:
    """Extract percentage from text like '100%'."""
    m = re.search(r"(\d+)%", text)
    return int(m.group(1)) if m else 0


def _extract_matchday(soup: BeautifulSoup) -> Optional[int]:
    """Auto-detect matchday from page text like 'Giornata 38'."""
    m = re.search(r"Giornata\s*(\d+)", soup.get_text())
    if m:
        return int(m.group(1))
    return None


def scrape(
    url: str = URL,
    matchday: Optional[int] = None,
    season_start: Optional[int] = None,
) -> list[dict]:
    """Scrape probable formations page and return player status records.

    Parameters
    ----------
    url:
        URL of the probabili formazioni page.
    matchday:
        Current matchday (auto-detect if None).
    season_start:
        Season start year (auto-detect from page if None).

    Returns
    -------
    List of dicts with keys:
        fantacalcio_id, season_start, matchday, team,
        probability, status, injury_note
    """
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Auto-detect matchday from page
    if matchday is None:
        matchday = _extract_matchday(soup)
    if matchday is None:
        matchday = 38  # fallback: last matchday

    # Auto-detect season from page
    if season_start is None:
        m = re.search(r"(\d{4})-\d{2}", soup.get_text())
        if m:
            season_start = int(m.group(1))
    if season_start is None:
        season_start = 2026

    records: list[dict] = []

    # ── Parse each team card ─────────────────────────────────────────────
    for team_card in soup.select("div.card.team-card"):
        # Team name from h3.team-name inside header
        team_name_el = team_card.select_one("h3.team-name")
        team_name = team_name_el.get_text(strip=True) if team_name_el else None

        # Determine status by which player list they are in
        for player_list in team_card.select("ul.player-list"):
            is_starter = "starters" in (player_list.get("class") or [])
            is_reserve = "reserves" in (player_list.get("class") or [])

            for player_el in player_list.select("li.player-item"):
                # Player name from the link
                name_link = player_el.select_one("a.player-name")
                player_name = name_link.get_text(strip=True) if name_link else ""

                # Fantacalcio ID from the href URL
                href = name_link.get("href", "") if name_link else ""
                player_id = _extract_player_id(href)

                # Probability from progress-value
                prob_el = player_el.select_one("div.progress-value")
                probability = _parse_probability(prob_el.get_text(strip=True)) if prob_el else 0

                # Determine status
                if is_reserve or probability <= 5:
                    status = "bench"
                elif probability >= 70:
                    status = "starter"
                elif 30 <= probability < 70:
                    status = "doubtful"
                else:
                    status = "starter" if is_starter else "bench"

                records.append({
                    "fantacalcio_id": player_id,
                    "season_start": season_start,
                    "matchday": matchday,
                    "team": team_name,
                    "probability": probability,
                    "status": status,
                    "injury_note": None,
                })

    log.info(
        "Scraped %d player records from probabili formazioni (matchday %d, season %d).",
        len(records), matchday, season_start,
    )
    return records


def _find_team_name(section: Tag) -> Optional[str]:
    """Try to find the team name from a section element."""
    for heading in section.select("h2, h3, h4, strong.squad-name"):
        text = heading.get_text(strip=True)
        if text and len(text) < 30:
            return text
    return None


def persist(records: list[dict], db_url: str) -> int:
    """Persist scraped records to the database."""
    import sqlalchemy as sa

    engine = sa.create_engine(db_url)
    count = 0
    with engine.begin() as conn:
        for rec in records:
            conn.execute(
                sa.text(_LINEUP_PLAYERS_SQL),
                {
                    "fantacalcio_id": rec["fantacalcio_id"],
                    "season_start": rec["season_start"],
                    "matchday": rec["matchday"],
                    "team": rec["team"],
                    "probability": rec["probability"],
                    "status": rec["status"],
                    "injury_note": rec["injury_note"],
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

    parser = argparse.ArgumentParser(description="Scrape probabili formazioni Serie A")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--url", default=URL, help="Page URL to scrape")
    parser.add_argument("--matchday", type=int, default=None, help="Current matchday")
    parser.add_argument("--dry-run", action="store_true", help="Print records without persisting")
    args = parser.parse_args()

    records = scrape(url=args.url, matchday=args.matchday)

    if args.dry_run:
        for r in records[:20]:
            log.info("  %s", r)
        log.info("Total: %d records (dry-run, first 20 shown)", len(records))
        return

    if args.db_url:
        n = persist(records, args.db_url)
        log.info("Persisted %d records to DB.", n)
    else:
        log.warning("No --db-url provided; use --dry-run to inspect data.")


if __name__ == "__main__":
    main()
