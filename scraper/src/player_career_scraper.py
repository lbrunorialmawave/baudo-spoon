from __future__ import annotations

"""Fetch a single player's most recent season stats from his FotMob career
history, in ANY league.

Architectural rule:
  LEAGUE_CATALOG determines what can be bulk-scraped; it does NOT decide
  what may be persisted. Uncatalogued leagues from careerHistory are
  first-class identities.

Rollout modes (PR6):
  shadow=True  — fetch + classify, never write to DB (Stage 1)
  normal       — persist uncatalogued leagues (Stage 2+)
"""

import json
import logging
import os
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT_SECONDS = 15
_REQUEST_DELAY_SECONDS = 1.0
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_ESTIMATED_MINUTES_PER_APPEARANCE = 70


@dataclass
class ForeignStatsResult:
    """Structured outcome of a foreign-player career fetch+persist batch."""

    candidates: int = 0
    fetched: int = 0
    persisted: int = 0
    unresolved: int = 0
    uncatalogued: int = 0
    skipped_invalid: int = 0
    skipped_other: int = 0
    rows_written: int = 0
    # Shadow-mode counters (Stage 1 rollout)
    would_persist: int = 0
    would_skip: int = 0
    shadow: bool = False
    invariant_ok: bool = True
    invariant_errors: list[str] = field(default_factory=list)

    @property
    def persistence_rate(self) -> float | None:
        if self.fetched == 0:
            return None
        # In shadow mode rate is based on would_persist
        if self.shadow:
            return self.would_persist / self.fetched
        return self.persisted / self.fetched

    def assert_conservation(self) -> None:
        errors: list[str] = []
        if self.candidates != self.fetched + self.unresolved:
            errors.append(
                f"candidates({self.candidates}) != fetched({self.fetched}) "
                f"+ unresolved({self.unresolved})"
            )
        if self.shadow:
            accounted = self.would_persist + self.would_skip + self.skipped_invalid + self.skipped_other
        else:
            accounted = self.persisted + self.skipped_invalid + self.skipped_other
        if self.fetched != accounted:
            errors.append(
                f"fetched({self.fetched}) != accounted({accounted}) "
                f"(shadow={self.shadow})"
            )
        self.invariant_errors = errors
        self.invariant_ok = not errors
        if errors:
            for msg in errors:
                log.error("foreign_stats invariant violated: %s", msg)

    def to_dict(self) -> dict[str, Any]:
        rate = self.persistence_rate
        d: dict[str, Any] = {
            "candidates": self.candidates,
            "fetched": self.fetched,
            "persisted": self.persisted,
            "unresolved": self.unresolved,
            "uncatalogued": self.uncatalogued,
            "skipped_invalid": self.skipped_invalid,
            "skipped_other": self.skipped_other,
            "rows_written": self.rows_written,
            "persistence_rate": None if rate is None else round(rate * 100, 1),
            "invariant_ok": self.invariant_ok,
            "invariant_errors": list(self.invariant_errors),
            "shadow": self.shadow,
        }
        if self.shadow:
            d["would_persist"] = self.would_persist
            d["would_skip"] = self.would_skip
        return d


def _slugify(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _fetch_next_data(player_fotmob_id: int, player_name: str) -> Optional[dict]:
    slug = _slugify(player_name)
    url = f"{FOTMOB_BASE_URL}/players/{player_fotmob_id}/overview/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.warning("player_career: fetch failed for %s (%d): %s", player_name, player_fotmob_id, exc)
        return None

    m = _NEXT_DATA_RE.search(html)
    if not m:
        log.warning("player_career: no __NEXT_DATA__ found for %s (%d)", player_name, player_fotmob_id)
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        log.warning("player_career: malformed __NEXT_DATA__ for %s (%d)", player_name, player_fotmob_id)
        return None


def _parse_season_start(season_name: str | None) -> int | None:
    if not season_name or not isinstance(season_name, str):
        return None
    m = re.match(r"^(\d{4})", season_name.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _select_season_entry(entries: list) -> Optional[dict]:
    """Pick most recent season by parsed start year — does not rely on entries[0]."""
    if not entries:
        return None
    scored: list[tuple[int, dict]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start = _parse_season_start(entry.get("seasonName"))
        if start is None:
            continue
        scored.append((start, entry))
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]


def _best_tournament_entry(season_entry: dict) -> Optional[dict]:
    tournaments = season_entry.get("tournamentStats", []) or []
    if not tournaments:
        return None

    def _score(t: dict) -> tuple[int, int, int]:
        if not isinstance(t, dict):
            return (0, 0, 0)
        name = (t.get("leagueName") or "").strip()
        try:
            apps = int(t.get("appearances") or 0)
        except (TypeError, ValueError):
            apps = 0
        has_name = 1 if name else 0
        catalogued = 1 if name in LEAGUE_CATALOG else 0
        return (has_name, catalogued, apps)

    return max(tournaments, key=_score)


def fetch_player_career_snapshot(
    player_fotmob_id: int, player_name: str
) -> Optional[dict[str, Any]]:
    data = _fetch_next_data(player_fotmob_id, player_name)
    if data is None:
        return None

    try:
        career = data["props"]["pageProps"]["data"]["careerHistory"]
        entries = career["careerItems"]["senior"]["seasonEntries"]
    except (KeyError, TypeError):
        log.warning(
            "player_career: unexpected payload shape for %s (%d)",
            player_name, player_fotmob_id,
        )
        return None
    if not entries or not isinstance(entries, list):
        return None

    season_entry = _select_season_entry(entries)
    if season_entry is None:
        return None

    tournament = _best_tournament_entry(season_entry)
    source = tournament or season_entry
    raw_league = tournament.get("leagueName") if tournament else None
    league_name = (raw_league or "").strip() or None

    try:
        appearances = int(source.get("appearances") or 0)
        goals = int(source.get("goals") or 0)
        assists = int(source.get("assists") or 0)
        rating_field = source.get("rating") or {}
        rating_value = (
            float(rating_field["rating"])
            if isinstance(rating_field, dict) and rating_field.get("rating") is not None
            else None
        )
    except (TypeError, ValueError):
        log.warning(
            "player_career: unparsable stats for %s (%d)",
            player_name, player_fotmob_id,
        )
        return None

    if appearances == 0 or not league_name:
        return None

    season_name = season_entry.get("seasonName", "")
    season_label = season_name.replace("/", "-") if "/" in season_name else season_name
    catalogued = league_name in LEAGUE_CATALOG

    minutes_estimate = appearances * _ESTIMATED_MINUTES_PER_APPEARANCE
    per90_factor = 90.0 / minutes_estimate if minutes_estimate else 0.0

    return {
        "player_fotmob_id": player_fotmob_id,
        "player_name": player_name,
        "league_name": league_name,
        "season_label": season_label,
        "appearances": appearances,
        "minutes_estimate": minutes_estimate,
        "rating": rating_value,
        "goals_per_90": round(goals * per90_factor, 3),
        "assists_per_90": round(assists * per90_factor, 3),
        "estimated": True,
        "catalogued": catalogued,
    }


def _persist_one_snapshot(session: Any, snap: dict[str, Any]) -> int:
    from .db import ingest_league_stats, normalize_league_name

    raw_name = (snap.get("league_name") or "").strip()
    if not raw_name:
        return 0

    league_name = normalize_league_name(raw_name)
    meta = LEAGUE_CATALOG.get(league_name)

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
    if snap.get("rating") is not None:
        stat_values.append(("rating", snap["rating"]))

    total = 0
    for stat_category, value in stat_values:
        total += ingest_league_stats(
            session=session,
            rows=[{**base_row, "value": value}],
            league_name=league_name,
            meta=meta,
            season_label=snap["season_label"],
            fotmob_season_id=-1,
            stat_type="players",
            stat_category=stat_category,
            commit=False,
        )
    return total


def fetch_and_persist_players(
    players: dict[int, str],
    db_url: str,
    delay_seconds: float = _REQUEST_DELAY_SECONDS,
    *,
    shadow: bool | None = None,
) -> ForeignStatsResult:
    """Fetch and optionally persist career snapshots.

    Args:
        shadow: When True, classify outcomes without writing to the DB
                (Stage 1 rollout). Defaults to FOREIGN_SHADOW_MODE env
                (truthy = shadow).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from .db import Base

    if shadow is None:
        shadow = os.environ.get("FOREIGN_SHADOW_MODE", "").lower() in ("1", "true", "yes")

    result = ForeignStatsResult(candidates=len(players), shadow=shadow)
    if not players:
        result.assert_conservation()
        return result

    engine = create_engine(db_url)
    if not shadow:
        Base.metadata.create_all(engine)

    total = len(players)
    with Session(engine) as session:
        for i, (player_id, name) in enumerate(players.items(), start=1):
            snapshot = fetch_player_career_snapshot(player_id, name)
            if snapshot is None:
                result.unresolved += 1
            else:
                result.fetched += 1
                if not snapshot.get("catalogued", True):
                    result.uncatalogued += 1

                if shadow:
                    # Stage 1: classify only
                    result.would_persist += 1
                    log.info(
                        "foreign_snapshot_shadow",
                        extra={
                            "event": "foreign_snapshot_shadow",
                            "player_fotmob_id": player_id,
                            "player_name": name,
                            "league_name": snapshot.get("league_name"),
                            "catalogued": snapshot.get("catalogued"),
                            "season": snapshot.get("season_label"),
                            "would_persist": True,
                        },
                    )
                else:
                    try:
                        rows = _persist_one_snapshot(session, snapshot)
                        session.commit()
                    except Exception as exc:  # noqa: BLE001
                        session.rollback()
                        log.error(
                            "foreign_snapshot_skipped",
                            extra={
                                "event": "foreign_snapshot_skipped",
                                "player_fotmob_id": player_id,
                                "reason": "persist_error",
                                "error": str(exc),
                            },
                        )
                        result.skipped_other += 1
                    else:
                        result.rows_written += rows
                        if rows > 0:
                            result.persisted += 1
                            log.info(
                                "foreign_snapshot_persisted",
                                extra={
                                    "event": "foreign_snapshot_persisted",
                                    "player_fotmob_id": player_id,
                                    "player_name": name,
                                    "league_name": snapshot.get("league_name"),
                                    "catalogued": snapshot.get("catalogued"),
                                    "season": snapshot.get("season_label"),
                                    "persisted": True,
                                    "rows": rows,
                                },
                            )
                        else:
                            result.skipped_invalid += 1

            if i < total:
                time.sleep(delay_seconds)

    result.assert_conservation()
    log.info(
        "player_career: shadow=%s candidates=%d fetched=%d persisted=%d "
        "would_persist=%d unresolved=%d uncatalogued=%d rate=%s invariant_ok=%s",
        result.shadow, result.candidates, result.fetched, result.persisted,
        result.would_persist, result.unresolved, result.uncatalogued,
        result.persistence_rate, result.invariant_ok,
    )
    return result
