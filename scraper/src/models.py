from __future__ import annotations

from dataclasses import dataclass

FOTMOB_BASE_URL = "https://www.fotmob.com"
SERIE_A = "Serie A"


@dataclass(frozen=True, slots=True)
class LeagueMeta:
    display_name: str
    comp_id: str
    slug: str
    file_stem: str
    # Used for /leagues/{comp_id}/stats/{stats_slug} and the deep-stats API
    stats_slug: str
    # ISO-3 country code for the Fotmob deep-stats API (e.g. "ITA")
    country_code: str


LEAGUE_CATALOG: dict[str, LeagueMeta] = {
    SERIE_A: LeagueMeta(
        display_name=SERIE_A,
        comp_id="55",
        slug="serie-a",
        file_stem="serie_a",
        stats_slug="serie",
        country_code="ITA",
    ),
    "Premier League": LeagueMeta(
        display_name="Premier League",
        comp_id="47",
        slug="premier-league",
        file_stem="premier_league",
        stats_slug="premier-league",
        country_code="ENG",
    ),
    "La Liga": LeagueMeta(
        display_name="La Liga",
        comp_id="87",
        slug="laliga",
        file_stem="la_liga",
        stats_slug="laliga",
        country_code="ESP",
    ),
    "Bundesliga": LeagueMeta(
        display_name="Bundesliga",
        comp_id="54",
        slug="bundesliga",
        file_stem="bundesliga",
        stats_slug="bundesliga",
        country_code="GER",
    ),
    "Ligue 1": LeagueMeta(
        display_name="Ligue 1",
        comp_id="53",
        slug="ligue-1",
        file_stem="ligue_1",
        stats_slug="ligue-1",
        country_code="FRA",
    ),
    # Serie B — lower tier than the top-5 leagues. Persisted so the
    # player-career fallback can give promoted sides a baseline, but the
    # MANTRA runner discounts its per-90 output (see runner.load_data) since
    # performance in the cadetteria does not translate 1:1 to Serie A.
    "Serie B": LeagueMeta(
        display_name="Serie B",
        comp_id="157",
        slug="serie-b",
        file_stem="serie_b",
        stats_slug="serie-b",
        country_code="ITA",
    ),
}
