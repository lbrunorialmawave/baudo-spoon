from __future__ import annotations

"""Fetch a single player's most recent season stats from his FotMob career
history, in ANY league — used to give a player newly transferred into Serie A
a real performance baseline (see ml/mantra/runner.py's cross-league fallback,
migration 018) instead of leaving him as a blank neo-arrivo.

Unlike player_profile_scraper.py (which batches an entire league's roster
through a warmed Selenium session), this targets a handful of specific
players — new-to-Serie-A signings identified by the caller — so a plain
HTTP GET per player (no browser) is enough: FotMob's player overview page
embeds a full __NEXT_DATA__ JSON payload server-side, same technique, just
without the browser-batching overhead that only pays off at whole-league
scale.

Data available this way is coarser than the bulk topstats.json league dump
used elsewhere: FotMob's careerHistory gives appearances/goals/assists/
rating per season+competition, but not exact minutes played or xG/xA. Minutes
(and therefore every per-90 rate) are approximated from appearances — see
_ESTIMATED_MINUTES_PER_APPEARANCE. Every returned snapshot carries
"estimated": True so callers/downstream data-health views can flag it as
lower-fidelity than a real scrape.
"""

import json
import logging
import re
import time
import unicodedata
import urllib.request
from typing import Any

from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT_SECONDS = 15
_REQUEST_DELAY_SECONDS = 1.0
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)

# FotMob's career-history entries give appearances, not minutes played.
# 70 min/appearance is a rough middle-ground between a full start (~90) and
# a substitute cameo (~20-30) — good enough for a pre-season fallback
# baseline, not a substitute for the real per-90 scrape once the player
# has actual Serie A minutes.
_ESTIMATED_MINUTES_PER_APPEARANCE = 70


def _slugify(name: str) -> str:
    """Convert a player name to a FotMob URL slug (e.g. "John Stones" -> "john-stones")."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _fetch_next_data(player_fotmob_id: int, player_name: str) -> dict | None:
    slug = _slugify(player_name)
    url = f"{FOTMOB_BASE_URL}/players/{player_fotmob_id}/overview/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning(
            "player_career: fetch failed for %s (%d): %s",
            player_name,
            player_fotmob_id,
            exc,
        )
        return None

    m = _NEXT_DATA_RE.search(html)
    if not m:
        log.warning(
            "player_career: no __NEXT_DATA__ found for %s (%d)",
            player_name,
            player_fotmob_id,
        )
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        log.warning(
            "player_career: malformed __NEXT_DATA__ for %s (%d)",
            player_name,
            player_fotmob_id,
        )
        return None


def _best_tournament_entry(season_entry: dict) -> dict | None:
    """Pick the most representative competition within a season entry.

    Prefers a top-flight league already in LEAGUE_CATALOG over cups/UCL
    (low sample size, not comparable to a domestic league season), breaking
    ties by appearances. Returns None if no catalogued league is present —
    the caller then falls back to the season-level aggregate across every
    competition.
    """
    candidates = [
        t
        for t in season_entry.get("tournamentStats", []) or []
        if t.get("leagueName") in LEAGUE_CATALOG
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: int(t.get("appearances") or 0))


def fetch_player_career_snapshot(
    player_fotmob_id: int, player_name: str
) -> dict[str, Any] | None:
    """Return the player's most recent season snapshot, from any league.

    Returns None if the fetch failed, the payload shape was unexpected, or
    the player has no recorded appearances in his most recent season entry.
    """
    data = _fetch_next_data(player_fotmob_id, player_name)
    if data is None:
        return None

    try:
        career = data["props"]["pageProps"]["data"]["careerHistory"]
        entries = career["careerItems"]["senior"]["seasonEntries"]
    except (KeyError, TypeError):
        log.warning(
            "player_career: unexpected payload shape for %s (%d)",
            player_name,
            player_fotmob_id,
        )
        return None
    if not entries:
        return None

    season_entry = entries[0]  # FotMob orders seasonEntries most-recent-first
    tournament = _best_tournament_entry(season_entry)
    source = tournament or season_entry
    league_name = tournament.get("leagueName") if tournament else None

    try:
        appearances = int(source.get("appearances") or 0)
        goals = int(source.get("goals") or 0)
        assists = int(source.get("assists") or 0)
        rating_field = source.get("rating") or {}
        rating_value = (
            float(rating_field["rating"])
            if rating_field.get("rating") is not None
            else None
        )
    except (TypeError, ValueError):
        log.warning(
            "player_career: unparsable stats for %s (%d)", player_name, player_fotmob_id
        )
        return None

    if appearances == 0:
        return None

    season_name = season_entry.get("seasonName", "")  # e.g. "2024/2025"
    season_label = season_name.replace("/", "-") if "/" in season_name else season_name

    minutes_estimate = appearances * _ESTIMATED_MINUTES_PER_APPEARANCE
    per90_factor = 90.0 / minutes_estimate if minutes_estimate else 0.0

    return {
        "player_fotmob_id": player_fotmob_id,
        "player_name": player_name,
        # Falls back to "Unknown" (not persisted — see persist_career_snapshots)
        # when the player's only recorded competition was a cup/UCL entry
        # not in LEAGUE_CATALOG.
        "league_name": league_name or "Unknown",
        "season_label": season_label,
        "appearances": appearances,
        "minutes_estimate": minutes_estimate,
        "rating": rating_value,
        "goals_per_90": round(goals * per90_factor, 3),
        "assists_per_90": round(assists * per90_factor, 3),
        "estimated": True,
    }


def _persist_one_snapshot(session: Any, snap: dict[str, Any]) -> int:
    """Upsert a single career snapshot into player_season_stats via the
    existing league-stats ingestion path (same table/schema the bulk
    scraper uses), so MANTRA's cross-league COALESCE fallback
    (migration 018) can read it with no further downstream changes.
    """
    from .db import ingest_league_stats

    meta = LEAGUE_CATALOG.get(snap["league_name"])
    if meta is None:
        log.warning(
            "player_career: skipping %s — league %r not in LEAGUE_CATALOG",
            snap["player_name"],
            snap["league_name"],
        )
        return 0

    base_row = {
        "entity_id": snap["player_fotmob_id"],
        "entity_name": snap["player_name"],
        "team_id": None,
        "team_name": "",
        "rank": None,
    }
    stat_values: list[tuple[str, float]] = [
        ("mins_played", snap["minutes_estimate"]),
        ("goals_per_90", snap["goals_per_90"]),
        ("goal_assist", snap["assists_per_90"]),
    ]
    if snap["rating"] is not None:
        stat_values.append(("rating", snap["rating"]))

    total = 0
    for stat_category, value in stat_values:
        total += ingest_league_stats(
            session=session,
            rows=[{**base_row, "value": value}],
            league_name=snap["league_name"],
            meta=meta,
            season_label=snap["season_label"],
            # Not scraped from a bulk topstats.json season link (no such ID
            # is exposed by the career-history payload) — -1 flags this row
            # as sourced from the targeted per-player fetch rather than the
            # league-wide scraper.
            fotmob_season_id=-1,
            stat_type="players",
            stat_category=stat_category,
        )
    return total


def fetch_and_persist_players(
    players: dict[int, str],
    db_url: str,
    delay_seconds: float = _REQUEST_DELAY_SECONDS,
) -> tuple[int, int]:
    """Fetch and persist career snapshots one player at a time.

    Persisting immediately after each fetch (rather than collecting
    everything and writing once at the end) means a request that gets cut
    short — a proxy timeout, a container restart — only loses the players
    not yet reached, not the ones already fetched: re-running (without
    force) naturally picks up where it left off, since already-covered
    players drop out of the caller's candidate query.

    Returns (fetched_count, persisted_rows_count).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from .db import Base

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    fetched = 0
    persisted = 0
    total = len(players)
    with Session(engine) as session:
        for i, (player_id, name) in enumerate(players.items(), start=1):
            snapshot = fetch_player_career_snapshot(player_id, name)
            if snapshot is not None:
                fetched += 1
                persisted += _persist_one_snapshot(session, snapshot)
            if i < total:
                time.sleep(delay_seconds)

    log.info(
        "player_career: fetched %d/%d players, %d rows persisted",
        fetched,
        total,
        persisted,
    )
    return fetched, persisted
