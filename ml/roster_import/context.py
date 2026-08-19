"""Runtime-only RosterContext.

Holds the matched rose of the user and of the other teams in the same
division for the duration of a request (or client session).  Nothing here
is persisted to PostgreSQL or Redis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Sequence

from ml.roster_import.matcher import (
    MatchQuality,
    MatchStatus,
    MatchedDivision,
    MatchedPlayer,
    MatchedTeam,
    match_workbook,
    CatalogPlayer,
)
from ml.roster_import.parser import ParsedWorkbook


@dataclass(frozen=True, slots=True)
class RosterContext:
    """In-memory context produced by a successful roster import + match.

    Identified by a short-lived ``context_id`` that the client echoes back
    on subsequent lineup / trades calls.  The server keeps the object only
    for the lifetime of the request (or a process-local cache with TTL if
    the deployment chooses to); it is never written to the database.
    """

    context_id: str
    source_filename: str | None
    divisions: tuple[MatchedDivision, ...]
    quality: MatchQuality

    # Optional: which team the user claimed as "mine" (set after selection).
    user_team_key: str | None = None
    """Format: ``"{sheet_name}::{team_name}"``."""

    def list_teams(
        self,
        *,
        division: str | None = None,
        include_empty: bool = False,
    ) -> list[tuple[str, str, int, int, bool]]:
        """Return ``(sheet_name, team_name, player_count, total_spent, is_empty)``."""
        out: list[tuple[str, str, int, int, bool]] = []
        for div in self.divisions:
            if division and div.sheet_name != division:
                continue
            for t in div.teams:
                if t.is_empty and not include_empty:
                    continue
                out.append(
                    (
                        div.sheet_name,
                        t.team_name,
                        len(t.players),
                        t.total_spent,
                        t.is_empty,
                    )
                )
        return out

    def get_team(self, sheet_name: str, team_name: str) -> MatchedTeam | None:
        for div in self.divisions:
            if div.sheet_name != sheet_name:
                continue
            for t in div.teams:
                if t.team_name == team_name:
                    return t
        return None

    def get_user_team(self) -> MatchedTeam | None:
        if not self.user_team_key:
            return None
        sheet, _, name = self.user_team_key.partition("::")
        return self.get_team(sheet, name)

    def teams_in_same_division(
        self,
        sheet_name: str,
        *,
        exclude_team: str | None = None,
    ) -> list[MatchedTeam]:
        """Teams selectable as opponent (same division only — plan D4)."""
        for div in self.divisions:
            if div.sheet_name != sheet_name:
                continue
            return [
                t
                for t in div.teams
                if not t.is_empty and t.team_name != exclude_team
            ]
        return []

    def with_user_team(self, sheet_name: str, team_name: str) -> RosterContext:
        """Return a new context with the claimed user team set."""
        key = f"{sheet_name}::{team_name}"
        if self.get_team(sheet_name, team_name) is None:
            raise ValueError(
                f"Team '{team_name}' not found in division '{sheet_name}'"
            )
        return RosterContext(
            context_id=self.context_id,
            source_filename=self.source_filename,
            divisions=self.divisions,
            quality=self.quality,
            user_team_key=key,
        )

    def unmatched_players(self) -> list[MatchedPlayer]:
        out: list[MatchedPlayer] = []
        for div in self.divisions:
            for t in div.teams:
                for p in t.players:
                    if p.status == MatchStatus.UNMATCHED:
                        out.append(p)
        return out

    def provisional_players(self) -> list[MatchedPlayer]:
        out: list[MatchedPlayer] = []
        for div in self.divisions:
            for t in div.teams:
                for p in t.players:
                    if p.status == MatchStatus.PROVISIONAL or p.needs_review:
                        out.append(p)
        return out


def build_roster_context(
    workbook: ParsedWorkbook,
    catalog: Sequence[CatalogPlayer],
    *,
    context_id: str | None = None,
) -> RosterContext:
    """Parse result + catalog → fully matched runtime context."""
    divisions, quality = match_workbook(workbook, catalog)
    return RosterContext(
        context_id=context_id or str(uuid.uuid4()),
        source_filename=workbook.source_filename,
        divisions=divisions,
        quality=quality,
    )
