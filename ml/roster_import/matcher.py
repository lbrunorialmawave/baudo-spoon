"""Match parsed rose players against the official Fantacalcio catalog.

The catalog is an in-memory sequence of :class:`CatalogPlayer` (typically
loaded from ``player_quotations`` + ``player_mantra_roles``).  Matching is
pure and side-effect free; results feed into :class:`RosterContext`.

Thresholds (plan D3):
- score >= 0.92          → auto
- 0.75 <= score < 0.92   → provisional (needs_review=True)
- score < 0.75           → unmatched
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from ml.data.name_matching import (
    AUTO_MATCH_THRESHOLD,
    REVIEW_MATCH_THRESHOLD,
    last_name_token,
    normalise_player_name,
    score_name_pair,
)
from ml.roster_import.parser import ParsedPlayer, ParsedTeam, ParsedWorkbook

log = logging.getLogger(__name__)


# ── Catalog & match result types ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CatalogPlayer:
    """Minimal reference record from the official listone / mantra roles."""

    fantacalcio_id: int
    name: str
    """Canonical display name from the listone."""

    team: str
    """Serie A team (or empty if unknown)."""

    role_classic: str | None = None
    """Classic role code (P/D/C/A) when available."""

    roles_mantra: tuple[str, ...] = ()
    """Mantra role codes (Por, Dc, …) when available."""

    # Pre-computed keys (filled by build_catalog_index)
    name_norm: str = field(default="", repr=False)
    last_name_norm: str = field(default="", repr=False)


class MatchStatus(str, Enum):
    AUTO = "auto"
    PROVISIONAL = "provisional"
    UNMATCHED = "unmatched"
    MANUAL = "manual"  # reserved for later manual resolution


@dataclass(frozen=True, slots=True)
class MatchedPlayer:
    """A parsed player after catalog matching."""

    parsed: ParsedPlayer
    status: MatchStatus
    score: float
    """Best similarity score in [0, 1]. 0 for unmatched."""

    catalog: CatalogPlayer | None = None
    """Matched catalog entry (None when unmatched)."""

    needs_review: bool = False
    """True when status is provisional or when multiple close candidates exist."""

    candidates: tuple[CatalogPlayer, ...] = ()
    """Alternative candidates near the top score (for UI disambiguation)."""


@dataclass(frozen=True, slots=True)
class MatchedTeam:
    team_name: str
    players: tuple[MatchedPlayer, ...]
    total_spent: int
    is_empty: bool

    @property
    def match_rate(self) -> float:
        if not self.players:
            return 1.0
        matched = sum(
            1
            for p in self.players
            if p.status in (MatchStatus.AUTO, MatchStatus.PROVISIONAL, MatchStatus.MANUAL)
        )
        return matched / len(self.players)


@dataclass(frozen=True, slots=True)
class MatchedDivision:
    sheet_name: str
    teams: tuple[MatchedTeam, ...]


@dataclass(frozen=True, slots=True)
class MatchQuality:
    total_players: int
    auto: int
    provisional: int
    unmatched: int
    match_rate: float
    """(auto + provisional + manual) / total_players."""

    by_team: dict[str, float] = field(default_factory=dict)


# ── Catalog index ────────────────────────────────────────────────────────────


def prepare_catalog(players: Iterable[CatalogPlayer]) -> list[CatalogPlayer]:
    """Return a new list with ``name_norm`` / ``last_name_norm`` filled."""
    prepared: list[CatalogPlayer] = []
    for p in players:
        n_norm = normalise_player_name(p.name)
        ln = last_name_token(n_norm)
        prepared.append(
            CatalogPlayer(
                fantacalcio_id=p.fantacalcio_id,
                name=p.name,
                team=p.team,
                role_classic=p.role_classic,
                roles_mantra=p.roles_mantra,
                name_norm=n_norm,
                last_name_norm=ln,
            )
        )
    return prepared


def _index_by_last_name(
    catalog: Sequence[CatalogPlayer],
) -> dict[str, list[CatalogPlayer]]:
    idx: dict[str, list[CatalogPlayer]] = defaultdict(list)
    for p in catalog:
        if p.last_name_norm:
            idx[p.last_name_norm].append(p)
        # Also index full name_norm for exact hits on short names
        if p.name_norm and p.name_norm != p.last_name_norm:
            idx[p.name_norm].append(p)
    return idx


# ── Core matching ────────────────────────────────────────────────────────────


def match_player(
    parsed: ParsedPlayer,
    catalog: Sequence[CatalogPlayer],
    *,
    last_name_index: dict[str, list[CatalogPlayer]] | None = None,
) -> MatchedPlayer:
    """Match a single parsed player against the catalog.

    Strategy (ordered):
    1. Exact normalised full-name hit.
    2. Exact last-name token hit (may produce multiple → disambiguate by score).
    3. Fuzzy score over last-name / full-name against the whole catalog
       (or the last-name bucket when the index is provided).
    """
    q_norm = normalise_player_name(parsed.name_clean)
    q_ln = last_name_token(q_norm)

    if not q_norm:
        return MatchedPlayer(
            parsed=parsed,
            status=MatchStatus.UNMATCHED,
            score=0.0,
        )

    # 1) Exact full-name
    exact = [c for c in catalog if c.name_norm == q_norm]
    if len(exact) == 1:
        return MatchedPlayer(
            parsed=parsed,
            status=MatchStatus.AUTO,
            score=1.0,
            catalog=exact[0],
        )
    if len(exact) > 1:
        # Ambiguous exact (rare) — treat as provisional with all candidates
        return MatchedPlayer(
            parsed=parsed,
            status=MatchStatus.PROVISIONAL,
            score=1.0,
            catalog=exact[0],
            needs_review=True,
            candidates=tuple(exact),
        )

    # 2) Exact last-name bucket
    candidates: list[CatalogPlayer] = []
    if last_name_index is not None and q_ln:
        candidates = list(last_name_index.get(q_ln, []))
    if not candidates and q_ln:
        candidates = [c for c in catalog if c.last_name_norm == q_ln]

    # 3) Fuzzy over candidates (or full catalog if bucket empty)
    search_pool = candidates if candidates else list(catalog)
    scored: list[tuple[float, CatalogPlayer]] = []
    for c in search_pool:
        s = score_name_pair(parsed.name_clean, c.name, use_last_name_only=True)
        if s >= REVIEW_MATCH_THRESHOLD:
            scored.append((s, c))

    if not scored:
        return MatchedPlayer(
            parsed=parsed,
            status=MatchStatus.UNMATCHED,
            score=0.0,
        )

    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]

    # Near-ties → provisional + expose alternatives
    alternatives = [c for s, c in scored[1:] if abs(s - best_score) < 0.05][:5]

    if best_score >= AUTO_MATCH_THRESHOLD and not alternatives:
        status = MatchStatus.AUTO
        needs_review = False
    elif best_score >= REVIEW_MATCH_THRESHOLD:
        status = MatchStatus.PROVISIONAL
        needs_review = True
    else:
        return MatchedPlayer(
            parsed=parsed,
            status=MatchStatus.UNMATCHED,
            score=best_score,
        )

    return MatchedPlayer(
        parsed=parsed,
        status=status,
        score=best_score,
        catalog=best,
        needs_review=needs_review,
        candidates=tuple(alternatives),
    )


def match_team(
    team: ParsedTeam,
    catalog: Sequence[CatalogPlayer],
    *,
    last_name_index: dict[str, list[CatalogPlayer]] | None = None,
) -> MatchedTeam:
    players = tuple(
        match_player(p, catalog, last_name_index=last_name_index)
        for p in team.players
    )
    return MatchedTeam(
        team_name=team.name,
        players=players,
        total_spent=team.total_spent,
        is_empty=team.is_empty,
    )


def match_workbook(
    workbook: ParsedWorkbook,
    catalog: Sequence[CatalogPlayer],
) -> tuple[tuple[MatchedDivision, ...], MatchQuality]:
    """Match every player in the workbook and return quality metrics."""
    prepared = prepare_catalog(catalog)
    idx = _index_by_last_name(prepared)

    divisions: list[MatchedDivision] = []
    total = auto = provisional = unmatched = 0
    by_team: dict[str, float] = {}

    for div in workbook.divisions:
        teams: list[MatchedTeam] = []
        for team in div.teams:
            mt = match_team(team, prepared, last_name_index=idx)
            teams.append(mt)
            by_team[f"{div.sheet_name}/{team.name}"] = mt.match_rate
            for mp in mt.players:
                total += 1
                if mp.status == MatchStatus.AUTO:
                    auto += 1
                elif mp.status == MatchStatus.PROVISIONAL:
                    provisional += 1
                else:
                    unmatched += 1
        divisions.append(
            MatchedDivision(sheet_name=div.sheet_name, teams=tuple(teams))
        )

    match_rate = (auto + provisional) / total if total else 1.0
    quality = MatchQuality(
        total_players=total,
        auto=auto,
        provisional=provisional,
        unmatched=unmatched,
        match_rate=match_rate,
        by_team=by_team,
    )
    log.info(
        "Matching complete: total=%d auto=%d provisional=%d unmatched=%d "
        "rate=%.1f%%",
        total,
        auto,
        provisional,
        unmatched,
        match_rate * 100,
    )
    return tuple(divisions), quality
