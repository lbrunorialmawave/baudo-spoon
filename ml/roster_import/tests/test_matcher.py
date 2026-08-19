"""Tests for roster name matching and RosterContext."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.roster_import import (
    CatalogPlayer,
    MatchStatus,
    ParsedPlayer,
    build_roster_context,
    match_player,
    parse_workbook,
    prepare_catalog,
)
from ml.data.name_matching import (
    AUTO_MATCH_THRESHOLD,
    REVIEW_MATCH_THRESHOLD,
    last_name_token,
    normalise_player_name,
    score_name_pair,
)

FIXTURE = Path(__file__).parent / "fixture_lido_di_ostia.xlsx"


# ── Synthetic catalog covering real names from the fixture ───────────────────

def _catalog() -> list[CatalogPlayer]:
    """Minimal catalog with names that appear in the Lido fixture."""
    raw = [
        # exact / easy
        (1, "De Gea", "Fiorentina", "P", ("Por",)),
        (2, "Calhanoglu", "Inter", "C", ("C", "T")),
        (3, "Hojlund", "Napoli", "A", ("A", "Pc")),
        (4, "McTominay", "Napoli", "C", ("M", "C")),
        (5, "Svilar", "Roma", "P", ("Por",)),
        (6, "Dimarco", "Inter", "D", ("Ds", "E")),
        (7, "Barella", "Inter", "C", ("C", "M")),
        (8, "Thuram", "Inter", "A", ("A",)),
        (9, "Leao", "Milan", "A", ("A", "W")),
        (10, "Maignan", "Milan", "P", ("Por",)),
        # ambiguous Martinez family
        (20, "Martinez Lautaro", "Inter", "A", ("A", "Pc")),
        (21, "Martinez J.", "Venezia", "A", ("A",)),  # "Martinez Jo." style
        (22, "Martinez L.", "Inter", "A", ("A",)),  # another abbreviation
        # compound surnames
        (30, "De Ketelaere", "Atalanta", "A", ("A", "T")),
        (31, "Di Lorenzo", "Napoli", "D", ("Dd", "B")),
        (32, "Milinkovic-Savic V.", "Torino", "P", ("Por",)),
        (33, "Kolo Muani", "Juventus", "A", ("A", "Pc")),
        # names that should stay unmatched
        (99, "Nonexistent Player", "Ghost FC", "C", ("C",)),
    ]
    return [
        CatalogPlayer(
            fantacalcio_id=fid,
            name=name,
            team=team,
            role_classic=role,
            roles_mantra=mantra,
        )
        for fid, name, team, role, mantra in raw
    ]


# ── Name helpers ─────────────────────────────────────────────────────────────


def test_normalise_and_last_name():
    assert normalise_player_name("Martinez Jo.") == "martinez jo"
    assert last_name_token("martinez jo") == "martinez"
    assert last_name_token(normalise_player_name("De Ketelaere")) == "de ketelaere"
    # "kolo" is not a compound prefix particle; behaviour matches import_quotations
    # (docstring example there was aspirational vs actual code).
    assert last_name_token(normalise_player_name("Kolo Muani")) == "muani"
    assert last_name_token(normalise_player_name("Randal Kolo Muani")) == "muani"


def test_score_name_pair_abbreviated():
    # Roster style "Martinez Jo." vs catalog "Martinez J."
    s = score_name_pair("Martinez Jo.", "Martinez J.")
    assert s >= REVIEW_MATCH_THRESHOLD

    s2 = score_name_pair("De Gea", "De Gea")
    assert s2 == 1.0 or s2 >= 0.99


# ── match_player ─────────────────────────────────────────────────────────────


def test_exact_match():
    cat = prepare_catalog(_catalog())
    parsed = ParsedPlayer(
        name_raw="De Gea", name_clean="De Gea", cost=34, row_index=2
    )
    m = match_player(parsed, cat)
    assert m.status == MatchStatus.AUTO
    assert m.score == 1.0
    assert m.catalog is not None
    assert m.catalog.fantacalcio_id == 1
    assert not m.needs_review


def test_abbreviated_martinez_provisional_or_auto():
    cat = prepare_catalog(_catalog())
    parsed = ParsedPlayer(
        name_raw="Martinez Jo.",
        name_clean="Martinez Jo.",
        cost=62,
        row_index=2,
    )
    m = match_player(parsed, cat)
    # Should hit one of the Martinez entries with high enough score
    assert m.status in (MatchStatus.AUTO, MatchStatus.PROVISIONAL)
    assert m.catalog is not None
    assert "martinez" in m.catalog.name.lower()
    # Multiple Martinez → likely needs_review
    if m.status == MatchStatus.PROVISIONAL:
        assert m.needs_review


def test_compound_surname():
    cat = prepare_catalog(_catalog())
    parsed = ParsedPlayer(
        name_raw="De Ketelaere",
        name_clean="De Ketelaere",
        cost=46,
        row_index=10,
    )
    m = match_player(parsed, cat)
    assert m.status == MatchStatus.AUTO
    assert m.catalog is not None
    assert m.catalog.fantacalcio_id == 30


def test_unmatched():
    cat = prepare_catalog(_catalog())
    parsed = ParsedPlayer(
        name_raw="Zlatan Ibrahimovic",
        name_clean="Zlatan Ibrahimovic",
        cost=1,
        row_index=99,
    )
    m = match_player(parsed, cat)
    assert m.status == MatchStatus.UNMATCHED
    assert m.catalog is None
    assert m.score < REVIEW_MATCH_THRESHOLD


def test_milinkovic_savic():
    cat = prepare_catalog(_catalog())
    parsed = ParsedPlayer(
        name_raw="Milinkovic-Savic V.",
        name_clean="Milinkovic-Savic V.",
        cost=20,
        row_index=3,
    )
    m = match_player(parsed, cat)
    assert m.status in (MatchStatus.AUTO, MatchStatus.PROVISIONAL)
    assert m.catalog is not None
    assert m.catalog.fantacalcio_id == 32


# ── RosterContext on real fixture (partial catalog) ──────────────────────────


def test_build_context_partial_catalog():
    wb = parse_workbook(FIXTURE)
    cat = _catalog()
    ctx = build_roster_context(wb, cat)

    assert ctx.context_id
    assert ctx.quality.total_players > 500
    # With a tiny catalog most players are unmatched, but the ones we know
    # should be matched.
    assert ctx.quality.auto + ctx.quality.provisional >= 5

    # list teams (exclude empty)
    teams = ctx.list_teams(include_empty=False)
    assert all(not empty for *_, empty in teams)
    assert len(teams) == 24  # 12 B + 12 C

    # same-division scoping
    opponents = ctx.teams_in_same_division("Divisione B", exclude_team="F.Q.F.C")
    assert all(t.team_name != "F.Q.F.C" for t in opponents)
    assert len(opponents) == 11

    # claim user team
    ctx2 = ctx.with_user_team("Divisione B", "F.Q.F.C")
    assert ctx2.user_team_key == "Divisione B::F.Q.F.C"
    user = ctx2.get_user_team()
    assert user is not None
    assert user.team_name == "F.Q.F.C"
    assert user.total_spent == 474


def test_with_user_team_unknown_raises():
    wb = parse_workbook(FIXTURE)
    ctx = build_roster_context(wb, _catalog())
    with pytest.raises(ValueError, match="not found"):
        ctx.with_user_team("Divisione B", "Nonexistent FC")


def test_match_rate_on_known_team():
    """When catalog covers every player of a small synthetic team, rate=1."""
    from ml.roster_import.parser import ParsedTeam

    team = ParsedTeam(
        name="Test FC",
        players=(
            ParsedPlayer("De Gea", "De Gea", 34, 2),
            ParsedPlayer("Barella", "Barella", 40, 3),
        ),
        total_spent=74,
        column_index=0,
    )
    from ml.roster_import.matcher import match_team

    mt = match_team(team, prepare_catalog(_catalog()))
    assert mt.match_rate == 1.0
    assert all(p.status == MatchStatus.AUTO for p in mt.players)
