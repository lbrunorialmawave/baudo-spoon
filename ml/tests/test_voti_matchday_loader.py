"""Regression tests for ``ml.data.voti_matchday_loader``.

These tests pin the surname-only matching path that recovers the voti JSON
``"Carnesecchi"`` against a DB row stored as ``"Carnesecchi Marco"``.

Before the fix, ``resolve_fantacalcio_id`` only fell back to a
``last_name_token`` scan when the *query* name had multiple tokens
(``token != nname``). When the voti JSON provides a single-token surname
(``"Zemura"``, ``"Buksa"``, …) ``nname == token`` and the fallback was
skipped, leaving 100% of players unmatched. The fix introduces a
dedicated surname index so single-token voti names resolve correctly.
"""

from __future__ import annotations

from ml.data import voti_matchday_loader as vml


def _name(n: str) -> str:
    return vml.normalise_player_name(n)


def _team(t: str) -> str:
    return vml.normalise_team(vml.apply_team_alias(t))


def _build_indices(rows: list[tuple[int, str, str]]) -> tuple[dict, dict]:
    """Build the (name_index, surname_index) pair from quotation rows.

    Each row is ``(fantacalcio_id, name, team)`` in the listone form
    ("Surname First", e.g. ``"Carnesecchi Marco"``). We mirror the
    production loader behaviour and ask ``last_name_token`` to index the
    leading token (the Italian listone convention).
    """
    name_index: dict[tuple[str, str], int] = {}
    surname_index: dict[tuple[str, str], int] = {}
    for fid, name, team in rows:
        nname = _name(name)
        nteam = _team(team)
        if not nname:
            continue
        name_index.setdefault((nname, nteam), fid)
        name_index.setdefault((nname, ""), fid)
        token = vml.last_name_token(nname, assume_surname_first=True)
        if token:
            surname_index.setdefault((token, nteam), fid)
            surname_index.setdefault((token, ""), fid)
    return name_index, surname_index


# ── Surname-only resolution (the bug we are fixing) ──────────────────────────


def test_resolves_surname_only_voti_against_full_name_quotation():
    # DB has "Carnesecchi Marco" (listone full name). The voti JSON for
    # the matchday report ships only the surname, "Carnesecchi".
    name_index, surname_index = _build_indices([
        (101, "Carnesecchi Marco", "Atalanta"),
    ])

    fid = vml.resolve_fantacalcio_id(
        "Carnesecchi", "Atalanta", name_index, surname_index,
    )
    assert fid == 101


def test_resolves_surname_only_voti_for_udinese_2025():
    # Mirrors the production failure: g38 voti for Udinese/Verona.
    name_index, surname_index = _build_indices([
        (210, "Zemura Jordan", "Udinese"),
        (211, "Buksa Adam", "Udinese"),
        (212, "Solet Oumar", "Udinese"),
        (213, "Kristensen Thomas", "Udinese"),
        (214, "Montipò Lorenzo", "Verona"),
    ])

    for voti_name, expected in [
        ("Zemura", 210),
        ("Buksa", 211),
        ("Solet", 212),
        ("Montipò", 214),
    ]:
        team = "Verona" if voti_name == "Montipò" else "Udinese"
        assert (
            vml.resolve_fantacalcio_id(
                voti_name, team, name_index, surname_index,
            )
            == expected
        ), f"failed for {voti_name!r}"


def test_resolves_surname_plus_initial_against_full_name_quotation():
    # Voti sometimes uses the "Surname X." disambiguator. The match must
    # still succeed against a DB row stored with the full name.
    name_index, surname_index = _build_indices([
        (310, "Kristensen Thomas", "Udinese"),
    ])

    fid = vml.resolve_fantacalcio_id(
        "Kristensen T.", "Udinese", name_index, surname_index,
    )
    assert fid == 310


# ── Exact full-name path (regression for the original behaviour) ─────────────


def test_exact_full_name_match_still_works():
    name_index, surname_index = _build_indices([
        (410, "De Ketelaere Charles", "Atalanta"),
    ])

    fid = vml.resolve_fantacalcio_id(
        "De Ketelaere Charles", "Atalanta", name_index, surname_index,
    )
    assert fid == 410


def test_exact_surname_match_still_works_when_listone_is_surname_only():
    # Some quotations are already stored surname-only. The direct path
    # should hit before the surname fallback is consulted.
    name_index, surname_index = _build_indices([
        (510, "Bellanova Raoul", "Atalanta"),
    ])

    fid = vml.resolve_fantacalcio_id(
        "Bellanova Raoul", "Atalanta", name_index, surname_index,
    )
    assert fid == 510


# ── Surname + team takes precedence over surname-only ───────────────────────


def test_team_match_preferred_over_no_team_for_shared_surname():
    # Two players with surname "Esposito", one per team. The voti side
    # carries the team so the team-qualified surname entry must win.
    name_index, surname_index = _build_indices([
        (610, "Esposito Salvatore", "Napoli"),
        (611, "Esposito Antonio", "Spezia"),
    ])

    assert (
        vml.resolve_fantacalcio_id(
            "Esposito S.", "Napoli", name_index, surname_index,
        )
        == 610
    )
    assert (
        vml.resolve_fantacalcio_id(
            "Esposito A.", "Spezia", name_index, surname_index,
        )
        == 611
    )


# ── Surname fallback is opt-in (the old index still works) ──────────────────


def test_surname_index_optional_keeps_backward_compatible_path():
    # The legacy index was a single dict keyed by (full name, team).
    # ``resolve_fantacalcio_id`` must keep returning ``None`` for the
    # surname-only case when no surname index is supplied, instead of
    # raising.
    legacy_index = {
        (_name("Carnesecchi Marco"), _team("Atalanta")): 101,
    }
    assert (
        vml.resolve_fantacalcio_id(
            "Carnesecchi", "Atalanta", legacy_index, None,
        )
        is None
    )


def test_unknown_player_still_unmatched():
    name_index, surname_index = _build_indices([
        (710, "Bellanova Raoul", "Atalanta"),
    ])
    assert (
        vml.resolve_fantacalcio_id(
            "Zemura", "Udinese", name_index, surname_index,
        )
        is None
    )


# ── last_name_token contract (the lower-level helper) ──────────────────────


def test_last_name_token_surname_first_italian_listone():
    # The Italian listone encodes names as "Surname Name". When the loader
    # asks ``last_name_token`` to assume that shape, the leading token wins
    # so we can build a (cognome) key compatible with the voti JSON.
    assert vml.last_name_token("zemura jordan", assume_surname_first=True) == "zemura"
    assert vml.last_name_token("carnesecchi marco", assume_surname_first=True) == "carnesecchi"
    assert (
        vml.last_name_token("de ketelaere charles", assume_surname_first=True)
        == "de ketelaere"
    )


def test_last_name_token_default_unchanged_for_first_last():
    # The default behaviour (FotMob "First Last" / English-style) must not
    # regress: "Adrian Benedyczak" → "benedyczak", still.
    assert vml.last_name_token("adrian benedyczak") == "benedyczak"
    assert vml.last_name_token("lautaro martinez") == "martinez"
    assert vml.last_name_token("charles de ketelaere") == "de ketelaere"
