"""Unit test per :mod:`scraper.gruppo_esperti`.

Nessuna rete coinvolta: `_parse_role_section` e `_resolve_cross_reference`
lavorano su testo semplice (lo stesso formato che `_split_sections` produce
dopo aver convertito i `<br>` del post in newline), quindi si può passare
direttamente il testo grezzo di un thread reale.

I due casi di test sono gli esempi reali forniti dall'utente: uno con il
breakdown numerico completo (Zappacosta) e uno il cui commento è un puro
rimando ad un'altra sezione del post (Bernasconi → "Vedi possibili
sorprese"), che è esattamente il caso che prima veniva perso.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import scraper.gruppo_esperti as gruppo_esperti
from scraper.gruppo_esperti import (
    _discover_from_forum_listing,
    _parse_role_section,
    _resolve_cross_reference,
)

_SECTION_TEXT = (
    "ZAPPACOSTA Davide (1992) Esterno, in bagarre per un posto da titolare.\n"
    "Titolarità 6/10 - Media voto 7/10 - Salute 6/10 - Bonus 7/10 - "
    "Consiglio Esperti 7/10 - TOTALE 33/50\n"
    "Da una parte una bella delusione (lato bonus), dall'altra una piacevole "
    "sorpresa (tanti voti senza mai infortunarsi).\n"
    "\n"
    "BERNASCONI Lorenzo (2003) Esterno sinistro\n"
    "Titolarità 7/10 - Media voto 7/10 - Salute 8/10 - Bonus 6/10 - "
    "Consiglio Esperti 7/10 - TOTALE 35/50\n"
    "Vedi possibili sorprese\n"
)


def test_parse_role_section_extracts_full_breakdown() -> None:
    players = _parse_role_section(_SECTION_TEXT, role="DEF", team="Atalanta", url="https://x.test/topic")

    assert len(players) == 2
    zappacosta = players[0]
    assert zappacosta.name == "ZAPPACOSTA Davide"
    assert zappacosta.birth_year == 1992
    assert zappacosta.titolarita == 6
    assert zappacosta.media_voto == 7
    assert zappacosta.salute == 6
    assert zappacosta.bonus_label == "Bonus"
    assert zappacosta.bonus_value == 7
    assert zappacosta.consiglio_esperti == 7
    assert zappacosta.totale == 33
    assert "delusione" in zappacosta.comment


def test_parse_role_section_handles_bernasconi_cross_reference_comment() -> None:
    players = _parse_role_section(_SECTION_TEXT, role="DEF", team="Atalanta", url="https://x.test/topic")

    bernasconi = players[1]
    assert bernasconi.name == "BERNASCONI Lorenzo"
    assert bernasconi.birth_year == 2003
    assert bernasconi.titolarita == 7
    assert bernasconi.totale == 35
    assert bernasconi.comment == "Vedi possibili sorprese"
    # Not resolved yet at this point — that's a separate pass in scrape_team.
    assert bernasconi.cross_reference_section is None


def test_resolve_cross_reference_matches_known_section() -> None:
    sections = {
        "POSSIBILI.SORPRESE": "Bernasconi potrebbe ritagliarsi spazio se Udogie dovesse fermarsi.",
    }

    marker, text = _resolve_cross_reference("Vedi possibili sorprese", sections)

    assert marker == "POSSIBILI.SORPRESE"
    assert text == sections["POSSIBILI.SORPRESE"]


def test_resolve_cross_reference_returns_none_for_plain_comment() -> None:
    marker, text = _resolve_cross_reference("Ottimo rendimento, tienilo stretto.", {})

    assert marker is None
    assert text is None


# ── _discover_from_forum_listing ────────────────────────────────────────────

# Structure verified against the live 2026/27 forum section listing
# (viewforum.php?f=199): team links use class="topictitle" and a relative
# href, alongside unrelated links (a sticky summary thread, a "mark read"
# control) that must NOT be mistaken for a team thread.
_FORUM_LISTING_HTML = """
<html><body>
<a href="./viewtopic.php?t=232729&sid=abc" class="topictitle">TOPIC RIASSUNTIVO SCHEDE SQUADRA 2026/27</a>
<a href="./viewtopic.php?t=232668&sid=abc" class="topictitle max-md:py-1">ATALANTA [TOPIC UNICO]</a>
<a href="./viewtopic.php?t=232676&sid=abc" class="topictitle max-md:py-1">INTER [TOPIC UNICO]</a>
<a href="./markread.php?f=199" class="something-else">Segna tutti come letti</a>
</body></html>
"""


def test_discover_from_forum_listing_extracts_team_urls(monkeypatch) -> None:
    monkeypatch.setattr(
        gruppo_esperti, "_get_soup",
        lambda url, session: BeautifulSoup(_FORUM_LISTING_HTML, "html.parser"),
    )

    teams = _discover_from_forum_listing("https://forum.gruppoesperti.it/viewforum.php?f=199", session=None)

    assert teams == {
        "ATALANTA": "https://forum.gruppoesperti.it/viewtopic.php?t=232668",
        "INTER": "https://forum.gruppoesperti.it/viewtopic.php?t=232676",
    }


def test_discover_team_topics_dispatches_to_forum_listing(monkeypatch) -> None:
    """A viewforum.php URL must go through the listing path, not the
    viewtopic.php first-post-links path (which would find no `div.post`
    on a listing page and raise)."""
    monkeypatch.setattr(
        gruppo_esperti, "_get_soup",
        lambda url, session: BeautifulSoup(_FORUM_LISTING_HTML, "html.parser"),
    )

    teams = gruppo_esperti.discover_team_topics("https://forum.gruppoesperti.it/viewforum.php?f=199")

    assert set(teams) == {"ATALANTA", "INTER"}
