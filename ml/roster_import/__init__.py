"""Roster import for Fantagazzetta multi-team Excel/CSV exports.

Parses rose (squads) into pure in-memory structures suitable for
``RosterContext`` (runtime-only, no DB persistence of current rosters).
"""

from .parser import (
    ParsedPlayer,
    ParsedTeam,
    ParsedDivision,
    ParsedWorkbook,
    parse_workbook,
    parse_bytes,
)
from .matcher import (
    CatalogPlayer,
    MatchStatus,
    MatchedPlayer,
    MatchedTeam,
    MatchedDivision,
    MatchQuality,
    match_player,
    match_team,
    match_workbook,
    prepare_catalog,
)
from .context import RosterContext, build_roster_context

__all__ = [
    # parser
    "ParsedPlayer",
    "ParsedTeam",
    "ParsedDivision",
    "ParsedWorkbook",
    "parse_workbook",
    "parse_bytes",
    # matcher
    "CatalogPlayer",
    "MatchStatus",
    "MatchedPlayer",
    "MatchedTeam",
    "MatchedDivision",
    "MatchQuality",
    "match_player",
    "match_team",
    "match_workbook",
    "prepare_catalog",
    # context
    "RosterContext",
    "build_roster_context",
]
