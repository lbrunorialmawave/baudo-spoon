"""Scraper for forum.gruppoesperti.it team analysis threads.

The forum publishes one staff-curated "[TOPIC UNICO]" thread per Serie A
team, linked from a per-season index topic. Each thread's *first post*
(edited by the team's staff account throughout the season — no login
required to read it) contains the starting XI, ballottaggi, and a full
per-player breakdown: a 1-10 "Consiglio Esperti" score plus a free-text
comment, grouped by role.

There is no API and the content is prose, not structured data, so parsing
relies on two anchors the site happens to use consistently across teams:

  1. Section headers are images whose *filename* encodes the section name
     (e.g. ".../aUAnrs2.png/PORTIERI"), so role-section boundaries are
     exact even though the visible page shows only a picture.
  2. Every player block contains a fixed-shape stat line:
         "Titolarità X/10 - Media voto X/10 - Salute X/10 - <label> X/10 -
          Consiglio Esperti X/10 - TOTALE NN/50"
     (<label> varies: "No Gol" for keepers, "Bonus" or "Porta inviolata"
     depending on the curator) which anchors each player regardless of the
     surrounding free text.

This is inherently heuristic: a curator who deviates from the template
causes that one player to be silently skipped, not mis-recorded. Verified
against Inter/Napoli/Milan/Roma for the 2025/26 season profiles at ~90%+
per-team coverage; expect occasional misses and revisit the regexes if a
team's coverage drops noticeably.

Usage:
    python -m scraper.gruppo_esperti --db-url postgresql://... --season-start 2025
    python -m scraper.gruppo_esperti --dry-run --team Inter
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

INDEX_URL = "https://forum.gruppoesperti.it/viewtopic.php?f=234&t=228046"
SOURCE = "gruppo_esperti"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
#: Seconds to wait between requests to the same forum, out of courtesy.
REQUEST_DELAY = 1.5

#: Sentinel matchday for "current season profile" (as opposed to a specific
#: matchday-scoped rating). Postgres UNIQUE constraints don't treat NULL as
#: equal to NULL, so a real value is needed here for re-scrapes to upsert
#: in place instead of accumulating duplicate rows.
PRESEASON_MATCHDAY = 0

ROLE_MARKERS: dict[str, str] = {
    "PORTIERI": "GK",
    "DIFENSORI": "DEF",
    "CENTROCAMPISTI": "MID",
    "ATTACCANTI": "FWD",
}

#: All section-marker filenames seen in the wild, used to know which
#: <img class="postimage"> tags delimit a new section while walking the post.
_ALL_MARKERS = set(ROLE_MARKERS) | {
    "ROSA", "RIGORISTI", "CALCI.PIAZZATI", "CONSIGLIATI", "SCONSIGLIATI",
    "POSSIBILI.SORPRESE", "PROSPETTO.PRIMAVERA", "COPYRIGHT", "GEPRESENTA",
    "PROBABILE.FORMAZIONE", "BALLOTTAGGI",
}

_STATS_LINE_RE = re.compile(
    r"Titolarit[àa’]\s*(\d+)\s*/\s*10\s*-\s*Media voto\s*(\d+)\s*/\s*10\s*-\s*"
    r"Salute\s*(\d+)\s*/\s*10\s*-\s*[^\d/\n]{2,30}?(\d+)\s*/\s*10\s*-\s*"
    r"Consiglio Esperti\s*(\d+)\s*/\s*10\s*-\s*TOTALE\s*(\d+)\s*/\s*50",
    re.IGNORECASE,
)

#: A player header line: "SURNAME Firstname (1997) role description".
#: Surnames are all-caps in the source and may be multi-word (DE VRIJ,
#: ZAMBO ANGUISSA); extended Latin ranges cover Balkan/Central-European
#: names (Ć, Đ, Š, Ž, ...).
_NAME_LINE_RE = re.compile(
    r"([A-ZÀ-ŽĀ-ſ][A-ZÀ-ŽĀ-ſ' \-]{1,40}?\s+"
    r"[A-ZÀ-Üa-zà-ÿĀ-ſ][A-Za-zà-ÿĀ-ſ'.\- ]{0,40}?)\s*"
    r"\(((?:19|20)\d{2})\)\s*([^\n]*)"
)

_TITLE_TEAM_RE = re.compile(r"^(.*?)\s*\[TOPIC UNICO\]", re.IGNORECASE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ScrapedPlayer:
    name: str
    role: str  # GK / DEF / MID / FWD
    team: str
    consiglio_esperti: int  # 1-10
    comment: str
    url: str


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def discover_team_topics(index_url: str = INDEX_URL, session: Optional[requests.Session] = None) -> dict[str, str]:
    """Parse the season index topic and return {team_name: topic_url}.

    Team names are read from each topic's own <title> tag ("INTER [TOPIC
    UNICO] - ...") rather than from the index page markup, since the index
    page's own team labels are only reliably present as image-filename
    hints and are missing for a few teams.
    """
    session = session or requests.Session()
    soup = _get_soup(index_url, session)
    first_post = soup.select_one("div.post")
    if first_post is None:
        raise RuntimeError(f"No post found on index page {index_url}")

    urls: list[str] = []
    for a in first_post.select("a"):
        if "topic unico" in a.get_text(strip=True).lower():
            href = a.get("href", "")
            url = requests.compat.urljoin(index_url, href).split("&sid=")[0]
            if url not in urls:
                urls.append(url)

    teams: dict[str, str] = {}
    for url in urls:
        time.sleep(REQUEST_DELAY)
        try:
            topic_soup = _get_soup(url, session)
        except requests.RequestException:
            log.exception("Failed to fetch topic %s while discovering team name", url)
            continue
        title = topic_soup.title.get_text(strip=True) if topic_soup.title else ""
        m = _TITLE_TEAM_RE.match(title)
        team = m.group(1).strip() if m else None
        if not team:
            log.warning("Could not determine team name for topic %s (title=%r)", url, title)
            continue
        teams[team] = url
    return teams


# ── Post parsing ─────────────────────────────────────────────────────────────


def _split_sections(content: Tag) -> dict[str, str]:
    """Split the first post's content into named sections.

    Section boundaries are <img class="postimage"> tags whose src ends in
    "/<SECTION_NAME>" (e.g. ".../aUAnrs2.png/PORTIERI"). <br> tags are
    turned into newlines so consecutive player blocks don't run together.
    """
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for el in content.descendants:
        if isinstance(el, Tag) and el.name == "img" and "postimage" in (el.get("class") or []):
            marker = el.get("src", "").rsplit("/", 1)[-1].upper()
            if marker in _ALL_MARKERS:
                current = marker
                sections.setdefault(current, [])
            continue
        if isinstance(el, Tag) and el.name == "br":
            if current:
                sections.setdefault(current, []).append("\n")
            continue
        if isinstance(el, NavigableString) and current:
            sections.setdefault(current, []).append(str(el))
    return {k: "".join(v) for k, v in sections.items()}


def _parse_role_section(text: str, role: str, team: str, url: str) -> list[ScrapedPlayer]:
    stats_matches = list(_STATS_LINE_RE.finditer(text))
    players: list[ScrapedPlayer] = []
    prev_end = 0
    for i, sm in enumerate(stats_matches):
        gap = text[prev_end:sm.start()]
        name_matches = list(_NAME_LINE_RE.finditer(gap))
        prev_end = sm.end()
        if not name_matches:
            log.debug("Skipping unmatched player header before stats line in %s (%s)", team, role)
            continue
        name = re.sub(r"\s+", " ", name_matches[-1].group(1)).strip()

        next_start = stats_matches[i + 1].start() if i + 1 < len(stats_matches) else len(text)
        commentary_block = text[sm.end():next_start]
        next_names = list(_NAME_LINE_RE.finditer(commentary_block))
        commentary = commentary_block[: next_names[-1].start()] if next_names else commentary_block
        commentary = re.sub(r"\n{2,}", "\n\n", commentary).strip()

        consiglio_esperti = int(sm.group(5))
        players.append(ScrapedPlayer(
            name=name, role=role, team=team,
            consiglio_esperti=consiglio_esperti,
            comment=commentary, url=url,
        ))
    return players


def scrape_team(team: str, url: str, session: Optional[requests.Session] = None) -> list[ScrapedPlayer]:
    """Scrape the first post of a single team's TOPIC UNICO thread."""
    session = session or requests.Session()
    soup = _get_soup(url, session)
    first_post = soup.select_one("div.post")
    if first_post is None:
        log.warning("No posts found for %s (%s)", team, url)
        return []
    content = first_post.select_one("div.content")
    if content is None:
        log.warning("First post has no content div for %s (%s)", team, url)
        return []

    sections = _split_sections(content)
    players: list[ScrapedPlayer] = []
    for marker, role in ROLE_MARKERS.items():
        section_text = sections.get(marker, "")
        if not section_text:
            log.warning("No %s section found for %s (%s)", marker, team, url)
            continue
        players.extend(_parse_role_section(section_text, role, team, url))

    log.info("Scraped %d players for %s", len(players), team)
    return players


def scrape(index_url: str = INDEX_URL, team_filter: Optional[str] = None) -> list[ScrapedPlayer]:
    """Scrape all (or one) team threads linked from the season index topic."""
    session = requests.Session()
    teams = discover_team_topics(index_url, session)
    if team_filter:
        teams = {t: u for t, u in teams.items() if t.lower() == team_filter.lower()}
        if not teams:
            raise ValueError(f"Team {team_filter!r} not found in index (available: {sorted(discover_team_topics(index_url, session))})")

    all_players: list[ScrapedPlayer] = []
    for team, url in teams.items():
        time.sleep(REQUEST_DELAY)
        try:
            all_players.extend(scrape_team(team, url, session))
        except requests.RequestException:
            log.exception("Failed to scrape %s (%s)", team, url)
    return all_players


# ── Name → fantacalcio_id matching ──────────────────────────────────────────

#: Known naming differences between the forum's team labels and
#: player_quotations.team (Fantacalcio listone naming).
_TEAM_ALIASES: dict[str, str] = {
    "VERONA": "HELLAS VERONA",
    "COMO": "COMO",
}

_FUZZY_MATCH_THRESHOLD = 0.75


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z ]", " ", s).upper()
    return re.sub(r"\s+", " ", s).strip()


def _load_quotations(conn, season_start: int) -> list[tuple[int, str, str]]:
    """Return [(fantacalcio_id, normalized_name, normalized_team), ...]."""
    import sqlalchemy as sa

    rows = conn.execute(
        sa.text(
            "SELECT fantacalcio_id, player_name, team FROM player_quotations "
            "WHERE season_start = :season"
        ),
        {"season": season_start},
    ).all()
    return [(r.fantacalcio_id, _normalize(r.player_name), _normalize(r.team)) for r in rows]


def _match_fantacalcio_id(
    name: str, team: str, quotations: list[tuple[int, str, str]],
) -> Optional[int]:
    norm_name = _normalize(name)
    norm_team = _TEAM_ALIASES.get(_normalize(team), _normalize(team))

    candidates = [q for q in quotations if q[2] == norm_team] or quotations

    best_id, best_score = None, 0.0
    for fid, q_name, _ in candidates:
        score = SequenceMatcher(None, norm_name, q_name).ratio()
        if score > best_score:
            best_id, best_score = fid, score
    if best_score >= _FUZZY_MATCH_THRESHOLD:
        return best_id
    return None


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_SQL = """
    INSERT INTO expert_ratings
        (player_id, source, expert_name, rating, comment, matchday, season_start, url, scraped_at)
    VALUES
        (:player_id, :source, :expert_name, :rating, :comment, :matchday, :season_start, :url, :scraped_at)
    ON CONFLICT (player_id, source, expert_name, matchday) DO UPDATE SET
        rating = EXCLUDED.rating,
        comment = EXCLUDED.comment,
        url = EXCLUDED.url,
        scraped_at = EXCLUDED.scraped_at
"""


def persist(players: list[ScrapedPlayer], db_url: str, season_start: int, expert_name: str = "gruppoesperti_staff") -> int:
    """Match players to fantacalcio_id and upsert ratings into expert_ratings.

    Players that can't be confidently matched to a known player_quotations
    row are skipped (and logged) rather than persisted with a guessed id.
    """
    import sqlalchemy as sa

    engine = sa.create_engine(db_url)
    count = 0
    unmatched = 0
    with engine.begin() as conn:
        quotations = _load_quotations(conn, season_start)
        if not quotations:
            log.warning(
                "No player_quotations rows for season_start=%s — cannot resolve any player_id. "
                "Import quotations first (ml.data.import_quotations).", season_start,
            )
        for p in players:
            fid = _match_fantacalcio_id(p.name, p.team, quotations)
            if fid is None:
                unmatched += 1
                log.debug("Unmatched player: %s (%s, %s)", p.name, p.team, p.role)
                continue
            conn.execute(
                sa.text(_UPSERT_SQL),
                {
                    "player_id": f"fc-{fid}",
                    "source": SOURCE,
                    "expert_name": expert_name,
                    "rating": max(1, min(5, round(p.consiglio_esperti / 2))),
                    "comment": p.comment,
                    "matchday": PRESEASON_MATCHDAY,
                    "season_start": season_start,
                    "url": p.url,
                    "scraped_at": datetime.now(timezone.utc),
                },
            )
            count += 1
    if unmatched:
        log.warning("%d/%d scraped players could not be matched to a fantacalcio_id.", unmatched, len(players))
    return count


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape team analyses from forum.gruppoesperti.it")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--index-url", default=INDEX_URL, help="Season index topic URL")
    parser.add_argument("--season-start", type=int, required=False, help="Season start year (e.g. 2025)")
    parser.add_argument("--team", help="Only scrape one team (by forum title, e.g. 'Inter')")
    parser.add_argument("--dry-run", action="store_true", help="Print records without persisting")
    args = parser.parse_args()

    players = scrape(index_url=args.index_url, team_filter=args.team)

    if args.dry_run:
        for p in players[:30]:
            log.info("  [%s] %s (%s) consiglio=%d/10", p.team, p.name, p.role, p.consiglio_esperti)
        log.info("Total: %d players scraped (dry-run, first 30 shown)", len(players))
        return

    if not args.db_url:
        log.warning("No --db-url provided; use --dry-run to inspect data.")
        return

    season_start = args.season_start or datetime.now().year
    n = persist(players, args.db_url, season_start)
    log.info("Persisted %d/%d ratings to DB.", n, len(players))


if __name__ == "__main__":
    main()
