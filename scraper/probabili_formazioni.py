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
import unicodedata
from difflib import SequenceMatcher
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
_FALLBACK_MATCH_THRESHOLD = 0.85


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


def _normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z ]", " ", text).upper()
    return re.sub(r"\s+", " ", text).strip()


def _load_quotations(conn, season_start: int) -> list[tuple[int, str, str]]:
    import sqlalchemy as sa

    rows = conn.execute(
        sa.text(
            "SELECT fantacalcio_id, player_name, team FROM player_quotations "
            "WHERE season_start = :season"
        ),
        {"season": season_start},
    ).all()
    return [(r.fantacalcio_id, _normalize_name(r.player_name), _normalize_name(r.team)) for r in rows]


def _candidate_variants(name: str) -> list[str]:
    norm_name = _normalize_name(name)
    variants = [norm_name]
    parts = norm_name.split()
    if parts:
        variants.extend([parts[0], parts[-1]])
    return [variant for variant in dict.fromkeys(variants) if variant]


def _exact_match_id(
    variants: list[str],
    candidates: list[tuple[int, str, str]],
) -> Optional[int]:
    for variant in variants:
        for fantacalcio_id, quotation_name, _ in candidates:
            if variant == quotation_name:
                return fantacalcio_id
            if quotation_name.startswith(variant + " ") or variant.startswith(quotation_name + " "):
                return fantacalcio_id

    return None


def _fuzzy_match_id(
    variants: list[str],
    candidates: list[tuple[int, str, str]],
) -> Optional[int]:

    best_id: Optional[int] = None
    best_score = 0.0
    for variant in variants:
        for fantacalcio_id, quotation_name, _ in candidates:
            score = SequenceMatcher(None, variant, quotation_name).ratio()
            if score > best_score:
                best_id = fantacalcio_id
                best_score = score

    if best_score >= _FALLBACK_MATCH_THRESHOLD:
        return best_id
    return None


def _best_match_id(
    variants: list[str],
    candidates: list[tuple[int, str, str]],
) -> Optional[int]:
    exact_match = _exact_match_id(variants, candidates)
    if exact_match is not None:
        return exact_match
    return _fuzzy_match_id(variants, candidates)


def _match_fantacalcio_id(name: str, team: str, quotations: list[tuple[int, str, str]]) -> Optional[int]:
    norm_team = _normalize_name(team)

    candidates = [q for q in quotations if q[2] == norm_team] or quotations
    return _best_match_id(_candidate_variants(name), candidates)


def _status_from_probability(probability: int, is_reserve: bool, is_starter: bool) -> str:
    if is_reserve or probability <= 5:
        return "bench"
    if probability >= 70:
        return "starter"
    if 30 <= probability < 70:
        return "doubtful"
    return "starter" if is_starter else "bench"


def _parse_player_item(
    player_el: Tag,
    team_name: Optional[str],
    season_start: int,
    matchday: int,
    is_reserve: bool,
    is_starter: bool,
) -> dict:
    name_link = player_el.select_one("a.player-name")
    player_name = name_link.get_text(strip=True) if name_link else ""
    href = name_link.get("href", "") if name_link else ""
    player_id = _extract_player_id(href)

    prob_el = player_el.select_one("div.progress-value")
    probability = _parse_probability(prob_el.get_text(strip=True)) if prob_el else 0

    return {
        "fantacalcio_id": player_id,
        "player_name": player_name,
        "season_start": season_start,
        "matchday": matchday,
        "team": team_name,
        "probability": probability,
        "status": _status_from_probability(probability, is_reserve, is_starter),
        "injury_note": None,
    }


def _parse_team_card(team_card: Tag, matchday: int, season_start: int) -> list[dict]:
    team_name_el = team_card.select_one("h3.team-name")
    team_name = team_name_el.get_text(strip=True) if team_name_el else None

    records: list[dict] = []
    for player_list in team_card.select("ul.player-list"):
        is_starter = "starters" in (player_list.get("class") or [])
        is_reserve = "reserves" in (player_list.get("class") or [])

        for player_el in player_list.select("li.player-item"):
            records.append(
                _parse_player_item(
                    player_el,
                    team_name,
                    season_start,
                    matchday,
                    is_reserve,
                    is_starter,
                )
            )

    return records


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
        records.extend(_parse_team_card(team_card, matchday, season_start))

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
        season_start = next((rec["season_start"] for rec in records if rec.get("season_start") is not None), None)
        quotations = _load_quotations(conn, season_start) if season_start is not None else []
        for rec in records:
            fantacalcio_id = rec.get("fantacalcio_id")
            if fantacalcio_id is None:
                fantacalcio_id = _match_fantacalcio_id(
                    rec.get("player_name", ""),
                    rec.get("team", ""),
                    quotations,
                )

            if fantacalcio_id is None:
                log.warning(
                    "Skipping probabili row without fantacalcio_id: player=%r team=%r matchday=%r season=%r",
                    rec.get("player_name"),
                    rec.get("team"),
                    rec.get("matchday"),
                    rec.get("season_start"),
                )
                continue

            conn.execute(
                sa.text(_LINEUP_PLAYERS_SQL),
                {
                    "fantacalcio_id": fantacalcio_id,
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
