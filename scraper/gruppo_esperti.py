"""Scraper for forum.gruppoesperti.it team analysis threads.

The forum publishes one staff-curated "[TOPIC UNICO]" thread per Serie A
team. Each thread's *first post* (edited by the team's staff account
throughout the season — no login required to read it) contains the
starting XI, ballottaggi, and a full per-player breakdown: a 1-10
"Consiglio Esperti" score plus a free-text comment, grouped by role.

Team threads are discovered from an "index" page, whose exact shape has
changed across seasons — the discovery step therefore supports two forms,
picked by ``INDEX_URL``'s own URL shape (see ``discover_team_topics``):

  1. ``viewtopic.php`` — a single curated topic whose first post links to
     each team's thread with anchor text containing "topic unico" (used
     through the 2025/26 season).
  2. ``viewforum.php`` — a forum *section* listing every team's thread
     directly as its own row (2026/27: the forum stopped curating a
     separate index topic and team threads live directly under the
     "Schede squadra e schede partita" section instead).

Whichever form is used, ``INDEX_URL`` needs re-checking every season —
open forum.gruppoesperti.it, find the current "Schede squadra e schede
partita" section under "AREA FANTACALCIO", and update the constant (or
pass ``--index-url`` / the admin endpoint's ``index_url`` param) if the
section id or index-topic id has changed.

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

#: 2026/27: the forum section "Schede squadra e schede partita" (f=199)
#: lists every team's [TOPIC UNICO] thread directly — no curated index
#: topic exists this season (unlike 2025/26's viewtopic.php-based index).
#: Re-verify every season; see the module docstring for where to look.
INDEX_URL = "https://forum.gruppoesperti.it/viewforum.php?f=199"
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

#: Non-role sections whose content is prose worth resolving a player comment
#: against (e.g. a comment that's just "Vedi possibili sorprese"). Excludes
#: COPYRIGHT/GEPRESENTA, which are boilerplate, not content a comment would
#: ever point to.
_CROSS_REFERENCE_MARKERS = {
    "ROSA", "RIGORISTI", "CALCI.PIAZZATI", "CONSIGLIATI", "SCONSIGLIATI",
    "POSSIBILI.SORPRESE", "PROSPETTO.PRIMAVERA", "PROBABILE.FORMAZIONE",
    "BALLOTTAGGI",
}

#: All section-marker filenames seen in the wild, used to know which
#: <img class="postimage"> tags delimit a new section while walking the post.
_ALL_MARKERS = set(ROLE_MARKERS) | _CROSS_REFERENCE_MARKERS | {"COPYRIGHT", "GEPRESENTA"}

#: Some curators write "Titolarità: 9/10" (colon after the label), others
#: "Titolarità 9/10" (no colon) — sometimes both styles appear in the same
#: post (e.g. Atalanta uses no-colon for Portieri/Difensori but colons for
#: Centrocampisti/Attaccanti, apparently from a later partial edit). Every
#: label is followed by an optional colon to tolerate both.
#:
#: The 4th stat's label varies by curator/role ("Bonus", "No Gol" for
#: keepers, "Porta inviolata"), so it's captured as its own group instead of
#: being consumed as an anonymous wildcard.
_STATS_LINE_RE = re.compile(
    r"Titolarit[àa’]\s*:?\s*(?P<titolarita>\d+)\s*/\s*10\s*-\s*"
    r"Media voto\s*:?\s*(?P<media_voto>\d+)\s*/\s*10\s*-\s*"
    r"Salute\s*:?\s*(?P<salute>\d+)\s*/\s*10\s*-\s*"
    r"(?P<bonus_label>[^\d/\n]{2,30}?)\s*:?\s*(?P<bonus_value>\d+)\s*/\s*10\s*-\s*"
    r"Consiglio Esperti\s*:?\s*(?P<consiglio_esperti>\d+)\s*/\s*10\s*-\s*"
    r"TOTALE\s*:?\s*(?P<totale>\d+)\s*/\s*50",
    re.IGNORECASE,
)

#: A player header line: "SURNAME Firstname (1997) role description".
#: Surname and firstname are captured separately (rather than as one blob)
#: because matching against player_quotations needs the surname alone —
#: Fantacalcio listoni list players by surname only ("Acerbi", "Sommer"),
#: and comparing "ACERBI Francesco" as a whole against "Acerbi" tanks a
#: SequenceMatcher ratio just from the length mismatch, regardless of how
#: correct the match is.
#: The surname group is greedy over whole all-caps *words* (not just
#: uppercase characters) so multi-word surnames (DE VRIJ, ZAMBO ANGUISSA,
#: LUIS HENRIQUE) are captured whole, stopping at the first word that isn't
#: fully uppercase. Extended Latin ranges cover Balkan/Central-European
#: names (Ć, Đ, Š, Ž, ...).
_NAME_LINE_RE = re.compile(
    r"(?P<surname>(?:[A-ZÀ-ŽĀ-ſ][A-ZÀ-ŽĀ-ſ'\-]*\s+)+)"
    r"(?P<firstname>[A-ZÀ-Üa-zà-ÿĀ-ſ][A-Za-zà-ÿĀ-ſ'.\- ]{0,40}?)\s*"
    r"\(((?:19|20)\d{2})\)\s*([^\n]*)"
)

_TITLE_TEAM_RE = re.compile(r"^(.*?)\s*\[TOPIC UNICO\]", re.IGNORECASE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ScrapedPlayer:
    name: str
    surname: str  # used for matching against player_quotations (surname-only)
    role: str  # GK / DEF / MID / FWD
    team: str
    consiglio_esperti: int  # 1-10
    comment: str
    url: str
    titolarita: Optional[int] = None  # 1-10
    media_voto: Optional[int] = None  # 1-10
    salute: Optional[int] = None  # 1-10
    bonus_label: Optional[str] = None  # "Bonus" / "No Gol" / "Porta inviolata"
    bonus_value: Optional[int] = None  # 1-10
    totale: Optional[int] = None  # 1-50
    birth_year: Optional[int] = None
    # Set when `comment` is just a pointer to another section of the post
    # (e.g. "Vedi possibili sorprese"), resolved against that section's text.
    cross_reference_section: Optional[str] = None
    cross_reference_text: Optional[str] = None


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _discover_from_forum_listing(listing_url: str, session: requests.Session) -> dict[str, str]:
    """Parse a forum *section* listing page and return {team_name: topic_url}.

    Used when the forum has no curated index topic (2026/27): every team's
    [TOPIC UNICO] thread is listed directly as its own row. The team name
    is read straight from the link's own text (via ``_TITLE_TEAM_RE``),
    unlike ``discover_team_topics``'s viewtopic.php path, which has to
    fetch each thread separately to read its <title> tag — the listing
    page already shows "TEAM [TOPIC UNICO]" as the link text itself.
    """
    soup = _get_soup(listing_url, session)
    teams: dict[str, str] = {}
    for a in soup.select("a"):
        m = _TITLE_TEAM_RE.match(a.get_text(strip=True))
        if not m:
            continue
        team = m.group(1).strip()
        href = a.get("href", "")
        url = requests.compat.urljoin(listing_url, href).split("&sid=")[0]
        teams.setdefault(team, url)
    return teams


def discover_team_topics(index_url: str = INDEX_URL, session: Optional[requests.Session] = None) -> dict[str, str]:
    """Return {team_name: topic_url}, discovered from ``index_url``.

    Dispatches on the URL's own shape: a ``viewforum.php`` section listing
    is parsed directly (see ``_discover_from_forum_listing``); anything
    else is treated as a curated ``viewtopic.php`` index topic whose first
    post links to each team's thread with anchor text containing "topic
    unico" (2025/26 and earlier).

    For the viewtopic.php path, team names are read from each topic's own
    <title> tag ("INTER [TOPIC UNICO] - ...") rather than from the index
    page markup, since the index page's own team labels are only reliably
    present as image-filename hints and are missing for a few teams.
    """
    session = session or requests.Session()

    if "viewforum.php" in index_url:
        return _discover_from_forum_listing(index_url, session)

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
        best_name_match = name_matches[-1]
        surname_words = re.sub(r"\s+", " ", best_name_match.group("surname")).strip().split(" ")
        # The "all-caps word" character classes span Unicode blocks (e.g.
        # Latin-1 Supplement) where upper/lowercase codepoints interleave,
        # so an accented lowercase leftover from the *previous* player's
        # comment (e.g. a trailing "però") can slip in as a leading token.
        # str.isupper() is Unicode-aware and catches what the regex ranges
        # can't; keep only the trailing run of genuinely all-caps words,
        # since a leaked fragment always precedes the real surname.
        uppercase_tail: list[str] = []
        for word in reversed(surname_words):
            if word.isupper():
                uppercase_tail.insert(0, word)
            else:
                break
        surname = " ".join(uppercase_tail) if uppercase_tail else " ".join(surname_words)
        firstname = re.sub(r"\s+", " ", best_name_match.group("firstname")).strip()
        name = f"{surname} {firstname}".strip()
        birth_year = int(best_name_match.group(3)) if best_name_match.group(3) else None

        next_start = stats_matches[i + 1].start() if i + 1 < len(stats_matches) else len(text)
        commentary_block = text[sm.end():next_start]
        next_names = list(_NAME_LINE_RE.finditer(commentary_block))
        commentary = commentary_block[: next_names[-1].start()] if next_names else commentary_block
        commentary = re.sub(r"\n{2,}", "\n\n", commentary).strip()

        players.append(ScrapedPlayer(
            name=name, surname=surname, role=role, team=team,
            consiglio_esperti=int(sm.group("consiglio_esperti")),
            comment=commentary, url=url,
            titolarita=int(sm.group("titolarita")),
            media_voto=int(sm.group("media_voto")),
            salute=int(sm.group("salute")),
            bonus_label=re.sub(r"\s+", " ", sm.group("bonus_label")).strip(" -") or None,
            bonus_value=int(sm.group("bonus_value")),
            totale=int(sm.group("totale")),
            birth_year=birth_year,
        ))
    return players


#: Matches a comment that's purely a pointer to another section instead of
#: repeating text already written elsewhere in the post, e.g. "Vedi
#: possibili sorprese" or "Vedi ROSA".
_CROSS_REFERENCE_RE = re.compile(r"\bVedi\s+([A-Za-zÀ-ÿ' ]+)", re.IGNORECASE)


def _resolve_cross_reference(comment: str, sections: dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    """If `comment` points at another section (e.g. "Vedi possibili
    sorprese"), resolve it against that section's text, already collected by
    `_split_sections` but otherwise only used to delimit role-section
    boundaries. Returns (section_marker, section_text), or (None, None) if
    the comment isn't a recognized pointer.
    """
    m = _CROSS_REFERENCE_RE.search(comment)
    if not m:
        return None, None
    normalized = re.sub(r"\s+", ".", m.group(1).strip().upper())
    for marker in _CROSS_REFERENCE_MARKERS:
        if normalized.startswith(marker):
            section_text = sections.get(marker, "").strip()
            return marker, (section_text or None)
    return None, None


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

    for p in players:
        p.cross_reference_section, p.cross_reference_text = _resolve_cross_reference(p.comment, sections)

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

#: Known naming differences between the forum's team labels (as derived from
#: each topic's <title>, e.g. "HELLAS VERONA") and player_quotations.team
#: (Fantacalcio listone naming, e.g. "Verona"). Keyed by the *forum* name.
_TEAM_ALIASES: dict[str, str] = {
    "HELLAS VERONA": "VERONA",
}

#: Surnames are compared after normalization, so this can be a tight
#: threshold — it only needs to absorb minor spelling/transliteration
#: differences (e.g. missing apostrophe), not first-name noise.
_FUZZY_MATCH_THRESHOLD = 0.85


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z ]", " ", s).upper()
    return re.sub(r"\s+", " ", s).strip()


def _load_quotations(conn, season_start: int) -> list[tuple[int, str, str, str]]:
    """Return [(fantacalcio_id, normalized_name, normalized_team, role), ...]."""
    import sqlalchemy as sa

    rows = conn.execute(
        sa.text(
            "SELECT fantacalcio_id, player_name, team, role FROM player_quotations "
            "WHERE season_start = :season"
        ),
        {"season": season_start},
    ).all()
    return [
        (r.fantacalcio_id, _normalize(r.player_name), _normalize(r.team), r.role)
        for r in rows
    ]


def _match_in(surname: str, candidates: list[tuple[int, str, str, str]]) -> Optional[int]:
    """Try to match ``surname`` (already normalized) within ``candidates``.

    Tier 1: exact surname match, or one is a whitespace-bounded prefix of
    the other (covers "Kelly" vs "Kelly L.", "Moreno" vs "Moreno Alb.").
    Tier 2: fuzzy fallback for minor spelling/transliteration differences.
    """
    for fid, q_name, _, _role in candidates:
        if surname == q_name:
            return fid
        if q_name.startswith(surname + " ") or surname.startswith(q_name + " "):
            return fid

    best_id, best_score = None, 0.0
    for fid, q_name, _, _role in candidates:
        score = SequenceMatcher(None, surname, q_name).ratio()
        if score > best_score:
            best_id, best_score = fid, score
    return best_id if best_score >= _FUZZY_MATCH_THRESHOLD else None


def _match_fantacalcio_id(
    surname: str, team: str, role: str, quotations: list[tuple[int, str, str, str]],
) -> Optional[int]:
    """Match a scraped player's surname to a fantacalcio_id.

    Matches on surname only, not the full name: Fantacalcio listoni list
    players by surname alone ("Acerbi", "Sommer"), occasionally with a
    disambiguating initial ("Kelly L.", "Moreno Alb."). Comparing the
    forum's full "SURNAME Firstname" against that would tank a
    SequenceMatcher ratio purely from length mismatch, independent of
    whether the player is actually the same.

    Three tiers, most confident first — mirrors the exact/relaxed-role/fuzzy
    escalation ``ml/data/import_quotations.py`` already uses to build
    ``player_id_map`` (a separate table, but the same underlying problem:
    resolving free-text names against ``player_quotations``):

      1. Same team (the common case, and disambiguates shared surnames
         across teams, e.g. two different "Sommer"s).
      2. Same role, any team — a transfer-window mismatch (forum thread's
         team hasn't caught up with player_quotations, or vice versa)
         leaves the team stale but role is a much safer signal to fall
         back on than nothing, since it's extracted independently on both
         sides (the forum's own role section vs. player_quotations.role).
      3. Whole listone, unrestricted — last resort.
    """
    norm_surname = _normalize(surname)
    norm_team = _TEAM_ALIASES.get(_normalize(team), _normalize(team))

    same_team = [q for q in quotations if q[2] == norm_team]
    fid = _match_in(norm_surname, same_team)
    if fid is not None:
        return fid

    same_role = [q for q in quotations if q[3] == role]
    fid = _match_in(norm_surname, same_role)
    if fid is not None:
        return fid

    return _match_in(norm_surname, quotations)


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_SQL = """
    INSERT INTO expert_ratings
        (player_id, source, expert_name, rating, comment, matchday, season_start, url, scraped_at,
         titolarita, media_voto, salute, bonus_label, bonus_value, totale, consiglio_esperti_raw,
         birth_year, cross_reference_section, cross_reference_text)
    VALUES
        (:player_id, :source, :expert_name, :rating, :comment, :matchday, :season_start, :url, :scraped_at,
         :titolarita, :media_voto, :salute, :bonus_label, :bonus_value, :totale, :consiglio_esperti_raw,
         :birth_year, :cross_reference_section, :cross_reference_text)
    ON CONFLICT (player_id, source, expert_name, matchday) DO UPDATE SET
        rating = EXCLUDED.rating,
        comment = EXCLUDED.comment,
        url = EXCLUDED.url,
        scraped_at = EXCLUDED.scraped_at,
        titolarita = EXCLUDED.titolarita,
        media_voto = EXCLUDED.media_voto,
        salute = EXCLUDED.salute,
        bonus_label = EXCLUDED.bonus_label,
        bonus_value = EXCLUDED.bonus_value,
        totale = EXCLUDED.totale,
        consiglio_esperti_raw = EXCLUDED.consiglio_esperti_raw,
        birth_year = EXCLUDED.birth_year,
        cross_reference_section = EXCLUDED.cross_reference_section,
        cross_reference_text = EXCLUDED.cross_reference_text
"""


def persist(
    players: list[ScrapedPlayer], db_url: str, season_start: Optional[int] = None,
    expert_name: str = "gruppoesperti_staff",
) -> tuple[int, Optional[int]]:
    """Match players to fantacalcio_id and upsert ratings into expert_ratings.

    Players that can't be confidently matched to a known player_quotations
    row are skipped (and logged) rather than persisted with a guessed id.

    If season_start is not given, it's resolved to the latest season present
    in player_quotations rather than derived from the current calendar date.
    A "current year" guess is unreliable right around the actual season
    boundary — e.g. forum threads for the 2025/26 season are still the
    active content through mid-2026, while ``datetime.now().year`` flips to
    2026 as soon as January does, well before that season's data exists.

    Returns
    -------
    (rows_persisted, resolved_season_start) — season_start is None only if
    player_quotations was completely empty and nothing could be resolved.
    """
    import sqlalchemy as sa

    engine = sa.create_engine(db_url)
    count = 0
    unmatched = 0
    with engine.begin() as conn:
        if season_start is None:
            season_start = conn.execute(
                sa.text("SELECT MAX(season_start) FROM player_quotations")
            ).scalar()
            if season_start is None:
                log.warning(
                    "player_quotations is empty — cannot resolve a season_start or any "
                    "player_id. Import quotations first (ml.data.import_quotations)."
                )
                return 0, None
            log.info("No --season-start given; resolved to latest available: %d", season_start)
        quotations = _load_quotations(conn, season_start)
        if not quotations:
            log.warning(
                "No player_quotations rows for season_start=%s — cannot resolve any player_id. "
                "Import quotations first (ml.data.import_quotations).", season_start,
            )
        for p in players:
            fid = _match_fantacalcio_id(p.surname, p.team, p.role, quotations)
            if fid is None:
                unmatched += 1
                log.warning(
                    "Unmatched player: %r (team=%r, role=%s) — no player_quotations row "
                    "for season_start=%s matched surname %r closely enough (own team or "
                    "otherwise). Check spelling/team in the listone, or whether the player "
                    "is missing from player_quotations for this season.",
                    p.name, p.team, p.role, season_start, p.surname,
                )
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
                    "titolarita": p.titolarita,
                    "media_voto": p.media_voto,
                    "salute": p.salute,
                    "bonus_label": p.bonus_label,
                    "bonus_value": p.bonus_value,
                    "totale": p.totale,
                    "consiglio_esperti_raw": p.consiglio_esperti,
                    "birth_year": p.birth_year,
                    "cross_reference_section": p.cross_reference_section,
                    "cross_reference_text": p.cross_reference_text,
                },
            )
            count += 1
    if unmatched:
        log.warning("%d/%d scraped players could not be matched to a fantacalcio_id.", unmatched, len(players))
    return count, season_start


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
            log.info(
                "  [%s] %s (%s, %s) tit=%s/10 media=%s/10 salute=%s/10 %s=%s/10 "
                "consiglio=%d/10 tot=%s/50%s",
                p.team, p.name, p.role, p.birth_year or "?",
                p.titolarita, p.media_voto, p.salute,
                p.bonus_label or "bonus", p.bonus_value,
                p.consiglio_esperti, p.totale,
                f" [ref: {p.cross_reference_section}]" if p.cross_reference_section else "",
            )
        log.info("Total: %d players scraped (dry-run, first 30 shown)", len(players))
        return

    if not args.db_url:
        log.warning("No --db-url provided; use --dry-run to inspect data.")
        return

    n, resolved_season = persist(players, args.db_url, args.season_start)
    log.info("Persisted %d/%d ratings to DB (season_start=%s).", n, len(players), resolved_season)


if __name__ == "__main__":
    main()
