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
from typing import Any, Optional, Sequence

from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG, LEAGUE_CATALOG_BY_COMP_ID

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
    # Season-resolution counters (PR4 extension of existing shadow/result)
    season_target_selected: int = 0
    season_previous_selected: int = 0
    season_latest_selected: int = 0
    season_no_valid: int = 0
    season_fallback_depth_total: int = 0
    season_fallback_depth_histogram: dict = field(default_factory=dict)

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
            "season_target_selected": self.season_target_selected,
            "season_previous_selected": self.season_previous_selected,
            "season_latest_selected": self.season_latest_selected,
            "season_no_valid": self.season_no_valid,
            "season_fallback_depth_total": self.season_fallback_depth_total,
            "season_fallback_depth_histogram": dict(self.season_fallback_depth_histogram),
        }
        if self.shadow:
            d["would_persist"] = self.would_persist
            d["would_skip"] = self.would_skip
        return d

    def record_season_resolution(self, snapshot: dict[str, Any] | None) -> None:
        """Accumulate season-resolution metrics from one fetch outcome."""
        if snapshot is None:
            self.season_no_valid += 1
            return
        reason = snapshot.get("selection_reason") or ""
        depth = int(snapshot.get("fallback_depth") or 0)
        if reason == REASON_TARGET_SEASON_SELECTED:
            self.season_target_selected += 1
        elif reason == REASON_PREVIOUS_SEASON_SELECTED:
            self.season_previous_selected += 1
        elif reason == REASON_LATEST_VALID_SELECTED:
            self.season_latest_selected += 1
        self.season_fallback_depth_total += depth
        hist = self.season_fallback_depth_histogram
        hist[depth] = hist.get(depth, 0) + 1


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


# ---------------------------------------------------------------------------
# Season-resolution domain (PR1)
# ---------------------------------------------------------------------------

# Reason codes for SeasonResolutionResult
REASON_TARGET_SEASON_SELECTED = "TARGET_SEASON_SELECTED"
REASON_PREVIOUS_SEASON_SELECTED = "PREVIOUS_SEASON_SELECTED"
REASON_NO_TARGET_SEASON = "NO_TARGET_SEASON"
REASON_TARGET_SEASON_INVALID = "TARGET_SEASON_INVALID"
REASON_NO_VALID_SEASON = "NO_VALID_SEASON"
REASON_SEASON_MALFORMED = "SEASON_MALFORMED"
REASON_LATEST_VALID_SELECTED = "LATEST_VALID_SELECTED"


@dataclass(frozen=True)
class ForeignPlayerCandidate:
    """One backfill/refresh unit: a player scoped to an explicit target season.

    The logical key is (player_fotmob_id, target_season_start). The same player
    may appear multiple times in a batch with different targets.
    """

    player_fotmob_id: int
    player_name: str
    target_season_start: int
    prediction_season_start: int | None = None

    def effective_prediction_season(self) -> int:
        return (
            self.prediction_season_start
            if self.prediction_season_start is not None
            else self.target_season_start
        )


@dataclass(frozen=True)
class CareerSeason:
    """Normalized season entry from FotMob careerHistory."""

    season_start: int
    season_label: str
    raw_entry: dict[str, Any]


@dataclass(frozen=True)
class SeasonResolutionPolicy:
    """Explicit policy controlling how a season is chosen from careerHistory.

    Modes (see plan §8):
      A — Current season refresh: target set, allow_previous_season_fallback=True
      B — Historical backfill:    target set, allow_previous_season_fallback=False
      C — Generic latest lookup:  target=None, allow_previous_season_fallback=False
    """

    target_season_start: int | None = None
    allow_previous_season_fallback: bool = False
    max_fallback_depth: int = 2
    require_positive_appearances: bool = True
    require_league: bool = True


@dataclass(frozen=True)
class SeasonResolutionResult:
    """Outcome of season resolution — always explains why a season was (or was not) chosen."""

    target_season_start: int | None
    selected_season_start: int | None
    fallback_depth: int
    reason: str
    entry: dict[str, Any] | None

    @property
    def selected(self) -> bool:
        return self.entry is not None


def _career_seasons_from_entries(entries: list) -> list[CareerSeason]:
    """Parse and order season entries deterministically by season_start ascending.

    Malformed / non-dict entries are dropped. Duplicate season_start values keep
    the first occurrence encountered (stable).
    """
    if not entries or not isinstance(entries, list):
        return []

    seen: set[int] = set()
    seasons: list[CareerSeason] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start = _parse_season_start(entry.get("seasonName"))
        if start is None:
            continue
        if start in seen:
            continue
        seen.add(start)
        label = (entry.get("seasonName") or "").strip()
        seasons.append(CareerSeason(season_start=start, season_label=label, raw_entry=entry))

    seasons.sort(key=lambda s: s.season_start)
    return seasons


@dataclass(frozen=True)
class CompetitionSnapshot:
    """Best competition chosen for an already-selected season entry."""

    league_name: str | None
    appearances: int
    goals: int
    assists: int
    rating: float | None
    catalogued: bool
    raw_tournament: dict[str, Any] | None
    selection_rank: tuple[int, int, int, int]
    competition_id: str | None = None  # FotMob leagueId; None for legacy/test data


def _competition_rank(t: dict) -> tuple[int, int, int, int]:
    """Explicit competition ranking (higher is better).

    Order (plan §11.2):
      1. has league name
      2. catalogued domestic/top league (tie-breaker, not a hard filter)
      3. appearances
      4. non-cup heuristic (name does not look like a cup) — only when
         FotMob metadata does not distinguish; we never invent classifications
         beyond a light name heuristic on common cup tokens.
    """
    if not isinstance(t, dict):
        return (0, 0, 0, 0)
    name = (t.get("leagueName") or "").strip()
    try:
        apps = int(t.get("appearances") or 0)
    except (TypeError, ValueError):
        apps = 0
    has_name = 1 if name else 0
    catalogued = 1 if name in LEAGUE_CATALOG else 0
    # Light cup heuristic from available text only — not a taxonomy invention.
    lower = name.lower()
    looks_like_cup = any(
        token in lower
        for token in ("cup", "copa", "coppa", "pokal", "trophy", "super cup", "supercup")
    )
    not_cup = 0 if looks_like_cup else 1
    return (has_name, catalogued, apps, not_cup)


def _best_tournament_entry(season_entry: dict) -> Optional[dict]:
    """Pick the best tournament dict from a season entry (legacy helper)."""
    result = resolve_competition(season_entry)
    return result.raw_tournament if result is not None else None


def resolve_competition(season_entry: dict) -> CompetitionSnapshot | None:
    """Competition resolver — independent of season selection.

    Given an already-chosen season entry, choose which competition best
    represents that season. Does not decide *which* season to use.
    """
    tournaments = season_entry.get("tournamentStats", []) or []
    if not tournaments:
        # Fall back to season-level aggregate fields when present.
        try:
            apps = int(season_entry.get("appearances") or 0)
        except (TypeError, ValueError):
            apps = 0
        if apps <= 0:
            return None
        return CompetitionSnapshot(
            league_name=None,
            appearances=apps,
            goals=int(season_entry.get("goals") or 0),
            assists=int(season_entry.get("assists") or 0),
            rating=None,
            catalogued=False,
            raw_tournament=None,
            selection_rank=(0, 0, apps, 0),
            competition_id=None,
        )

    best = max(
        (t for t in tournaments if isinstance(t, dict)),
        key=_competition_rank,
        default=None,
    )
    if best is None:
        return None

    name = (best.get("leagueName") or "").strip() or None
    competition_id = str(best.get("leagueId") or "").strip() or None
    try:
        apps = int(best.get("appearances") or 0)
    except (TypeError, ValueError):
        apps = 0
    try:
        goals = int(best.get("goals") or 0)
    except (TypeError, ValueError):
        goals = 0
    try:
        assists = int(best.get("assists") or 0)
    except (TypeError, ValueError):
        assists = 0
    rating_field = best.get("rating") or {}
    rating_value = (
        float(rating_field["rating"])
        if isinstance(rating_field, dict) and rating_field.get("rating") is not None
        else None
    )
    catalogued = bool(
        (competition_id is not None and competition_id in LEAGUE_CATALOG_BY_COMP_ID)
        or (competition_id is None and name is not None and name in LEAGUE_CATALOG)
    )
    return CompetitionSnapshot(
        league_name=name,
        appearances=apps,
        goals=goals,
        assists=assists,
        rating=rating_value,
        catalogued=catalogued,
        raw_tournament=best,
        selection_rank=_competition_rank(best),
        competition_id=competition_id,
    )


def _is_season_entry_usable(
    season_entry: dict,
    *,
    require_positive_appearances: bool = True,
    require_league: bool = True,
) -> bool:
    """Return True if the season has a competition snapshot that passes quality gates.

    Uses the same competition selection as the rest of the pipeline so that
    season resolution and later snapshot normalization stay consistent.
    """
    comp = resolve_competition(season_entry)
    if comp is None:
        return False
    if require_positive_appearances and comp.appearances <= 0:
        return False
    if require_league and not comp.league_name:
        return False
    return True


def resolve_season(
    entries: list,
    policy: SeasonResolutionPolicy | None = None,
) -> SeasonResolutionResult:
    """Select a season entry according to an explicit policy.

    Never uses entries[0] or bare max() without policy. Always returns a
    SeasonResolutionResult that records target, selected season, fallback depth
    and a reason code.
    """
    policy = policy or SeasonResolutionPolicy()
    seasons = _career_seasons_from_entries(entries)

    if not seasons:
        return SeasonResolutionResult(
            target_season_start=policy.target_season_start,
            selected_season_start=None,
            fallback_depth=0,
            reason=REASON_NO_VALID_SEASON if entries else REASON_NO_VALID_SEASON,
            entry=None,
        )

    def _usable(cs: CareerSeason) -> bool:
        return _is_season_entry_usable(
            cs.raw_entry,
            require_positive_appearances=policy.require_positive_appearances,
            require_league=policy.require_league,
        )

    # --- Mode C / no target: latest valid season (or latest if validity not required) ---
    if policy.target_season_start is None:
        # Walk from newest to oldest; stop at first usable when we care about validity.
        ordered_desc = list(reversed(seasons))
        for depth, cs in enumerate(ordered_desc):
            if not policy.require_positive_appearances and not policy.require_league:
                # Pure latest-by-year (legacy-compatible selection of the max entry)
                return SeasonResolutionResult(
                    target_season_start=None,
                    selected_season_start=cs.season_start,
                    fallback_depth=0,
                    reason=REASON_LATEST_VALID_SELECTED,
                    entry=cs.raw_entry,
                )
            if _usable(cs):
                return SeasonResolutionResult(
                    target_season_start=None,
                    selected_season_start=cs.season_start,
                    fallback_depth=depth,
                    reason=REASON_LATEST_VALID_SELECTED,
                    entry=cs.raw_entry,
                )
            if not policy.allow_previous_season_fallback:
                # Only the absolute latest was considered and it is unusable
                break
            if depth >= policy.max_fallback_depth:
                break
        return SeasonResolutionResult(
            target_season_start=None,
            selected_season_start=None,
            fallback_depth=0,
            reason=REASON_NO_VALID_SEASON,
            entry=None,
        )

    # --- Mode A / B: explicit target ---
    target = policy.target_season_start
    by_start = {cs.season_start: cs for cs in seasons}

    if target not in by_start:
        if not policy.allow_previous_season_fallback:
            return SeasonResolutionResult(
                target_season_start=target,
                selected_season_start=None,
                fallback_depth=0,
                reason=REASON_NO_TARGET_SEASON,
                entry=None,
            )
        # Target absent + fallback enabled: walk all seasons below the target.
        previous_from_absent = [cs for cs in seasons if cs.season_start < target]
        previous_from_absent.sort(key=lambda s: s.season_start, reverse=True)
        for depth, cs in enumerate(previous_from_absent, start=1):
            if depth > policy.max_fallback_depth:
                break
            if _usable(cs):
                return SeasonResolutionResult(
                    target_season_start=target,
                    selected_season_start=cs.season_start,
                    fallback_depth=depth,
                    reason=REASON_PREVIOUS_SEASON_SELECTED,
                    entry=cs.raw_entry,
                )
        return SeasonResolutionResult(
            target_season_start=target,
            selected_season_start=None,
            fallback_depth=0,
            reason=REASON_NO_VALID_SEASON,
            entry=None,
        )

    target_cs = by_start[target]
    if _usable(target_cs):
        return SeasonResolutionResult(
            target_season_start=target,
            selected_season_start=target,
            fallback_depth=0,
            reason=REASON_TARGET_SEASON_SELECTED,
            entry=target_cs.raw_entry,
        )

    # Target exists but is not usable
    if not policy.allow_previous_season_fallback:
        return SeasonResolutionResult(
            target_season_start=target,
            selected_season_start=None,
            fallback_depth=0,
            reason=REASON_TARGET_SEASON_INVALID,
            entry=None,
        )

    # Walk previous seasons (lower season_start), up to max_fallback_depth
    previous = [cs for cs in seasons if cs.season_start < target]
    previous.sort(key=lambda s: s.season_start, reverse=True)  # newest previous first

    for depth, cs in enumerate(previous, start=1):
        if depth > policy.max_fallback_depth:
            break
        if _usable(cs):
            return SeasonResolutionResult(
                target_season_start=target,
                selected_season_start=cs.season_start,
                fallback_depth=depth,
                reason=REASON_PREVIOUS_SEASON_SELECTED,
                entry=cs.raw_entry,
            )

    return SeasonResolutionResult(
        target_season_start=target,
        selected_season_start=None,
        fallback_depth=0,
        reason=REASON_NO_VALID_SEASON,
        entry=None,
    )


def _select_season_entry(entries: list) -> Optional[dict]:
    """Backward-compatible selector: pick the highest season_start entry.

    Preserves pre-PR1 behaviour used by fetch_player_career_snapshot (no
    usability filter here — validity is still checked after competition
    resolution). New callers should use resolve_season() with an explicit
    SeasonResolutionPolicy.
    """
    result = resolve_season(
        entries,
        SeasonResolutionPolicy(
            target_season_start=None,
            allow_previous_season_fallback=False,
            require_positive_appearances=False,
            require_league=False,
        ),
    )
    return result.entry


def _default_season_policy(
    target_season_start: int | None,
    season_policy: SeasonResolutionPolicy | None,
) -> SeasonResolutionPolicy:
    """Build the effective policy for a fetch call.

    - Explicit season_policy wins.
    - Otherwise: target set → Mode B (historical, no silent fallback).
      target None → Mode C (latest usable only, no walk-back) for BC.
    """
    if season_policy is not None:
        # Allow caller to override target via the dedicated kwarg if policy left it None
        if target_season_start is not None and season_policy.target_season_start is None:
            return SeasonResolutionPolicy(
                target_season_start=target_season_start,
                allow_previous_season_fallback=season_policy.allow_previous_season_fallback,
                max_fallback_depth=season_policy.max_fallback_depth,
                require_positive_appearances=season_policy.require_positive_appearances,
                require_league=season_policy.require_league,
            )
        return season_policy
    return SeasonResolutionPolicy(
        target_season_start=target_season_start,
        allow_previous_season_fallback=False,
        max_fallback_depth=2,
        require_positive_appearances=True,
        require_league=True,
    )


def fetch_player_career_snapshot(
    player_fotmob_id: int,
    player_name: str,
    *,
    target_season_start: int | None = None,
    prediction_season_start: int | None = None,
    season_policy: SeasonResolutionPolicy | None = None,
) -> Optional[dict[str, Any]]:
    """Fetch and normalise a single player's career snapshot.

    New optional kwargs (PR2) make the call season-aware:

    - ``target_season_start``: preferred source season (e.g. 2024 for historical backfill).
    - ``prediction_season_start``: season the stats will be used for (lineage).
      Defaults to ``target_season_start`` when provided, else to the selected source.
    - ``season_policy``: full control over fallback behaviour. When omitted a safe
      default is used (no previous-season fallback) so existing callers keep the
      pre-PR2 semantics.

    Returns None when no usable snapshot can be resolved.
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
            player_name, player_fotmob_id,
        )
        return None
    if not entries or not isinstance(entries, list):
        return None

    policy = _default_season_policy(target_season_start, season_policy)
    resolution = resolve_season(entries, policy)
    if not resolution.selected or resolution.entry is None:
        log.info(
            "player_career: no usable season for %s (%d) target=%s reason=%s",
            player_name,
            player_fotmob_id,
            policy.target_season_start,
            resolution.reason,
        )
        return None

    season_entry = resolution.entry
    comp = resolve_competition(season_entry)
    if comp is None:
        return None
    league_name = comp.league_name
    competition_id = comp.competition_id
    appearances = comp.appearances
    goals = comp.goals
    assists = comp.assists
    rating_value = comp.rating

    # Defence-in-depth: resolver already applied the same gates, but keep the
    # explicit check so a future policy change cannot silently persist junk.
    if appearances == 0 or not league_name:
        return None

    season_name = season_entry.get("seasonName", "")
    season_label = season_name.replace("/", "-") if "/" in season_name else season_name
    catalogued = bool(
        (competition_id is not None and competition_id in LEAGUE_CATALOG_BY_COMP_ID)
        or (competition_id is None and league_name in LEAGUE_CATALOG)
    )

    source_season_start = resolution.selected_season_start
    # prediction defaults: explicit kwarg → target → selected source
    pred_season = (
        prediction_season_start
        if prediction_season_start is not None
        else (
            policy.target_season_start
            if policy.target_season_start is not None
            else source_season_start
        )
    )

    minutes_estimate = appearances * _ESTIMATED_MINUTES_PER_APPEARANCE
    per90_factor = 90.0 / minutes_estimate if minutes_estimate else 0.0

    return {
        "player_fotmob_id": player_fotmob_id,
        "player_name": player_name,
        "league_name": league_name,
        "competition_id": competition_id,
        "season_label": season_label,
        "source_season_start": source_season_start,
        "prediction_season_start": pred_season,
        "selection_reason": resolution.reason,
        "fallback_depth": resolution.fallback_depth,
        "appearances": appearances,
        "minutes_estimate": minutes_estimate,
        "rating": rating_value,
        "goals_per_90": round(goals * per90_factor, 3),
        "assists_per_90": round(assists * per90_factor, 3),
        "estimated": True,
        "catalogued": catalogued,
    }


def _persist_one_snapshot(session: Any, snap: dict[str, Any]) -> int:
    """Persist one normalised career snapshot.

    ``fotmob_season_id=-1`` is the documented sentinel for foreign
    careerHistory rows that are not produced by a bulk league scrape
    (see migration 025 comment and plan §18.5).
    """
    from .db import ingest_league_stats, normalize_league_name

    raw_name = (snap.get("league_name") or "").strip()
    if not raw_name:
        return 0

    league_name = normalize_league_name(raw_name)
    comp_id_from_snap = snap.get("competition_id")
    if comp_id_from_snap:
        meta = LEAGUE_CATALOG_BY_COMP_ID.get(comp_id_from_snap)
    else:
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

    lineage = {
        "source_season_start": snap.get("source_season_start"),
        "prediction_season_start": snap.get("prediction_season_start"),
        "selection_reason": snap.get("selection_reason"),
        "fallback_depth": snap.get("fallback_depth"),
    }

    total = 0
    for stat_category, value in stat_values:
        total += ingest_league_stats(
            session=session,
            rows=[{**base_row, "value": value}],
            league_name=league_name,
            meta=meta,
            season_label=snap["season_label"],
            fotmob_season_id=-1,  # sentinel: foreign careerHistory snapshot
            stat_type="players",
            stat_category=stat_category,
            commit=False,
            **lineage,
        )
    return total


def fetch_and_persist_players(
    players: "dict[int, str] | Sequence[ForeignPlayerCandidate]",
    db_url: str,
    delay_seconds: float = _REQUEST_DELAY_SECONDS,
    *,
    shadow: bool | None = None,
    target_season_start: int | None = None,
    prediction_season_start: int | None = None,
    season_policy: SeasonResolutionPolicy | None = None,
) -> ForeignStatsResult:
    """Fetch and optionally persist career snapshots.

    Accepts either:
      - ``Sequence[ForeignPlayerCandidate]`` (preferred, PR4): each item carries
        its own ``target_season_start`` so multi-season batches keep the
        (player, target) relation end-to-end;
      - ``dict[int, str]`` (legacy): player_fotmob_id → name. A single global
        ``target_season_start`` / ``season_policy`` may still be applied.

    Collapsing a multi-season batch into ``dict[player_id] = name`` is forbidden
    when targets differ — callers must use ``ForeignPlayerCandidate``.
    """
    if shadow is None:
        shadow = os.environ.get("FOREIGN_SHADOW_MODE", "").lower() in ("1", "true", "yes")

    candidates: list[ForeignPlayerCandidate]
    if isinstance(players, dict):
        if target_season_start is None:
            candidates = [
                ForeignPlayerCandidate(
                    player_fotmob_id=pid,
                    player_name=name,
                    target_season_start=0,  # sentinel → no explicit target
                    prediction_season_start=prediction_season_start,
                )
                for pid, name in players.items()
            ]
        else:
            candidates = [
                ForeignPlayerCandidate(
                    player_fotmob_id=pid,
                    player_name=name,
                    target_season_start=target_season_start,
                    prediction_season_start=prediction_season_start,
                )
                for pid, name in players.items()
            ]
    else:
        candidates = list(players)

    result = ForeignStatsResult(candidates=len(candidates), shadow=shadow)
    if not candidates:
        result.assert_conservation()
        return result

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from .db import Base

    engine = create_engine(db_url)
    if not shadow:
        Base.metadata.create_all(engine)

    total = len(candidates)
    with Session(engine) as session:
        for i, cand in enumerate(candidates, start=1):
            cand_target = (
                None if cand.target_season_start == 0 else cand.target_season_start
            )
            if cand_target is not None:
                cand_pred = cand.effective_prediction_season()
            else:
                cand_pred = (
                    cand.prediction_season_start
                    if cand.prediction_season_start is not None
                    else prediction_season_start
                )

            cand_policy = season_policy
            if cand_policy is None and cand_target is not None:
                cand_policy = SeasonResolutionPolicy(
                    target_season_start=cand_target,
                    allow_previous_season_fallback=True,
                    max_fallback_depth=1,
                    require_positive_appearances=True,
                    require_league=True,
                )

            snapshot = fetch_player_career_snapshot(
                cand.player_fotmob_id,
                cand.player_name,
                target_season_start=cand_target,
                prediction_season_start=cand_pred,
                season_policy=cand_policy,
            )
            result.record_season_resolution(snapshot)

            if snapshot is None:
                result.unresolved += 1
            else:
                result.fetched += 1
                if not snapshot.get("catalogued", True):
                    result.uncatalogued += 1

                if shadow:
                    result.would_persist += 1
                    log.info(
                        "foreign_snapshot_shadow",
                        extra={
                            "event": "foreign_snapshot_shadow",
                            "player_fotmob_id": cand.player_fotmob_id,
                            "player_name": cand.player_name,
                            "league_name": snapshot.get("league_name"),
                            "catalogued": snapshot.get("catalogued"),
                            "season": snapshot.get("season_label"),
                            "target_season_start": cand_target,
                            "source_season_start": snapshot.get("source_season_start"),
                            "prediction_season_start": snapshot.get(
                                "prediction_season_start"
                            ),
                            "selection_reason": snapshot.get("selection_reason"),
                            "fallback_depth": snapshot.get("fallback_depth"),
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
                                "player_fotmob_id": cand.player_fotmob_id,
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
                                    "player_fotmob_id": cand.player_fotmob_id,
                                    "player_name": cand.player_name,
                                    "league_name": snapshot.get("league_name"),
                                    "catalogued": snapshot.get("catalogued"),
                                    "season": snapshot.get("season_label"),
                                    "target_season_start": cand_target,
                                    "source_season_start": snapshot.get(
                                        "source_season_start"
                                    ),
                                    "prediction_season_start": snapshot.get(
                                        "prediction_season_start"
                                    ),
                                    "selection_reason": snapshot.get("selection_reason"),
                                    "fallback_depth": snapshot.get("fallback_depth"),
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
        "would_persist=%d unresolved=%d uncatalogued=%d "
        "target_selected=%d previous_selected=%d no_valid=%d rate=%s invariant_ok=%s",
        result.shadow, result.candidates, result.fetched, result.persisted,
        result.would_persist, result.unresolved, result.uncatalogued,
        result.season_target_selected, result.season_previous_selected,
        result.season_no_valid,
        result.persistence_rate, result.invariant_ok,
    )
    return result
