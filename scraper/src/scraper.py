from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .config import settings
from .driver import get_managed_driver
from .match_stats_batch import fetch_matches_batch, stats_from_next_data
from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG, SERIE_A, LeagueMeta
from .parser import (
    create_team_rows,
    extract_possession,
    extract_stat_sections,
    parse_match_link,
)

_FINISHED_STATUSES = ("FT", "HT", "AET", "PEN")

log = logging.getLogger(__name__)


class FotMobMatchStatsScraper:
    """Fetch match stats from FotMob."""

    def __init__(
        self,
        leagues: str | Iterable[str] = SERIE_A,
        seasons: str | int | Iterable[str | int] | None = None,
        output_dir: Path = settings.output_dir,
    ) -> None:
        self.leagues = self._normalize_leagues(leagues)
        self.seasons = self._normalize_seasons(seasons)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        on_round_complete: Callable[[str, str, pd.DataFrame], None] | None = None,
    ) -> dict[tuple[str, str], tuple[pd.DataFrame, Path]]:
        """Scrape all configured leagues/seasons and persist results as CSV."""
        outputs: dict[tuple[str, str], tuple[pd.DataFrame, Path]] = {}

        with get_managed_driver() as driver:
            driver.get(FOTMOB_BASE_URL)
            time.sleep(2)

            for league_name in self.leagues:
                meta = LEAGUE_CATALOG[league_name]

                # If no season provided, we scrape the current one (e.g. 2023-2024 or 2024-2025).
                # For simplicity, if not provided we just use a default season string.
                # FotMob requires the exact season string like "2023-2024".
                seasons_to_scrape = (
                    self.seasons if self.seasons else [self._get_current_season()]
                )

                for season in seasons_to_scrape:
                    result = self._scrape_season_fast(
                        driver, meta, league_name, season, on_round_complete
                    )
                    if result is not None:
                        outputs[(league_name, season)] = result

        return outputs

    def _scrape_season_fast(
        self,
        driver: Any,
        meta: LeagueMeta,
        league_name: str,
        season: str,
        on_round_complete: Callable[[str, str, pd.DataFrame], None] | None,
    ) -> tuple[pd.DataFrame, Path] | None:
        """Two-phase scrape: (1) cheap fixture discovery per round via DOM,
        (2) ONE batched parallel fetch pass for all match stats, replacing
        the old per-match driver.get()+click+scroll loop.

        This is the fix for the ~1h runtime: previously every single match
        page was fully rendered and interacted with sequentially. Match
        stats already exist as JSON inside every match page's
        __NEXT_DATA__ blob, so we fetch that directly and in parallel via
        the browser's authenticated fetch(), batched to stay polite.
        """
        log.info("Scraping league=%s season=%s", league_name, season)
        total_rounds = 38  # Defaulting to 38 for major leagues

        # Phase 1 — fixture discovery (unchanged logic, still DOM-based
        # since it's cheap: 38 page loads regardless of match count).
        rounds_matches: dict[int, list[dict[str, Any]]] = {}
        for round_num in range(1, total_rounds + 1):
            log.info(
                "Discovering Round %d/%d for %s",
                round_num,
                total_rounds,
                meta.display_name,
            )
            try:
                rounds_matches[round_num] = self._scrape_matches_for_round(
                    driver, meta, season, round_num
                )
            except Exception as exc:
                log.error("Error discovering Round %d: %s", round_num, exc)
                rounds_matches[round_num] = []

        # Phase 2 — collect every finished match that needs stats, then
        # fetch all of them in one batched, parallel pass.
        finished_urls: list[str] = [
            match["url"]
            for matches in rounds_matches.values()
            for match in matches
            if match["status"] in _FINISHED_STATUSES
        ]
        total_matches = sum(len(m) for m in rounds_matches.values())
        log.info(
            "[%s %s] fixtures discovered: %d matches (%d finished, need stats)",
            league_name,
            season,
            total_matches,
            len(finished_urls),
        )

        stats_by_url = fetch_matches_batch(driver, finished_urls)
        ok = sum(1 for v in stats_by_url.values() if v)
        log.info(
            "[%s %s] batch stats fetch complete: %d/%d matches",
            league_name,
            season,
            ok,
            len(finished_urls),
        )

        # Phase 3 — assemble rows round by round, preserving the existing
        # on_round_complete streaming/incremental-DB-write behaviour.
        all_results: list[dict[str, Any]] = []
        for round_num in range(1, total_rounds + 1):
            round_results: list[dict[str, Any]] = []
            for match in rounds_matches.get(round_num, []):
                home_row, away_row = create_team_rows(match, round_num)

                groups = stats_by_url.get(match["url"])
                if groups:
                    stats_data = stats_from_next_data(groups)
                    for section_stats in stats_data.values():
                        for stat_name, values in section_stats.items():
                            home_row[stat_name] = values[0]
                            away_row[stat_name] = values[1]

                round_results.append(home_row)
                round_results.append(away_row)

            if round_results:
                all_results.extend(round_results)
                if on_round_complete is not None:
                    on_round_complete(league_name, season, pd.DataFrame(round_results))

        if not all_results:
            log.warning("No results found for %s %s", league_name, season)
            return None

        df = pd.DataFrame(all_results)
        output_path = self._output_file(meta.file_stem, season)
        df.to_csv(output_path, index=False, encoding="utf-8")
        log.info("Wrote %d rows → %s", len(df), output_path)
        return df, output_path

    # ------------------------------------------------------------------
    # LEGACY / fallback path — kept for reference and as an escape hatch
    # if FotMob ever changes __NEXT_DATA__ shape and batch fetching breaks.
    # Not used by run() anymore; _scrape_season_fast() above replaced it.
    # ------------------------------------------------------------------

    def _scrape_season(
        self, driver: Any, meta: LeagueMeta, season: str
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        total_rounds = 38  # Defaulting to 38 for major leagues

        for round_num in range(1, total_rounds + 1):
            log.info(
                "Starting Round %d/%d for %s",
                round_num,
                total_rounds,
                meta.display_name,
            )

            try:
                round_results = self._get_matches_with_stats(
                    driver, meta, season, round_num
                )
                if not round_results:
                    log.debug("No matches found in round %d.", round_num)
                yield round_num, round_results
            except Exception as exc:
                log.error("Error scraping Round %d: %s", round_num, exc)
                yield round_num, []

    def _get_matches_with_stats(
        self, driver: Any, meta: LeagueMeta, season: str, round_num: int
    ) -> list[dict[str, Any]]:
        matches = self._scrape_matches_for_round(driver, meta, season, round_num)
        if not matches:
            return []

        results: list[dict[str, Any]] = []

        for i, match in enumerate(matches, 1):
            log.debug(
                "Scraping match %d/%d: %s vs %s",
                i,
                len(matches),
                match["home"],
                match["away"],
            )
            home_row, away_row = create_team_rows(match, round_num)

            # Only scrape stats if match is finished
            if match["status"] in ("FT", "HT", "AET", "PEN"):
                stats = self._scrape_match_stats(driver, match["url"])

                # Flatten stats into rows
                for section_stats in stats.values():
                    for stat_name, values in section_stats.items():
                        # values[0] is Home, values[1] is Away
                        home_row[stat_name] = values[0]
                        away_row[stat_name] = values[1]

            results.append(home_row)
            results.append(away_row)

        return results

    def _scrape_matches_for_round(
        self, driver: Any, meta: LeagueMeta, season: str, round_num: int
    ) -> list[dict[str, Any]]:
        url_round = round_num - 1
        url = f"{FOTMOB_BASE_URL}/leagues/{meta.comp_id}/fixtures/{meta.slug}?group=by-round&season={season}&round={url_round}"
        log.debug("Navigating to matches URL: %s", url)

        try:
            driver.get(url)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            matches: list[dict[str, Any]] = []
            match_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/matches/']")

            for link in match_links:
                try:
                    match_url = link.get_attribute("href")
                    match_data = parse_match_link(link, match_url)
                    if match_data:
                        matches.append(match_data)
                except Exception as exc:
                    log.debug("Skipping unparseable match link: %s", exc)

            self._assign_dates(driver, matches)
            return matches

        except Exception as exc:
            log.warning(
                "Error occurred fetching matches for round %d: %s", round_num, exc
            )
            return []

    def _scrape_match_stats(self, driver: Any, match_url: str) -> dict[str, Any]:
        stats_data: dict[str, Any] = {}
        try:
            if not match_url.startswith("http"):
                match_url = f"{FOTMOB_BASE_URL}{match_url}"

            driver.get(match_url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(1)

            self._click_stats_tab(driver, wait)

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            extract_possession(driver, stats_data)
            extract_stat_sections(driver, stats_data)

            return {k: v for k, v in stats_data.items() if v}

        except Exception as exc:
            log.debug("Error scraping match stats for %s: %s", match_url, exc)
            return {}

    def _click_stats_tab(self, driver: Any, wait: WebDriverWait) -> None:
        try:
            stats_tab = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Stats')]")
                )
            )
            stats_tab.click()
            time.sleep(1.5)
        except Exception as exc:
            log.debug("Could not click Stats tab via wait: %s", exc)
            try:
                stats_tab = driver.find_element(By.XPATH, "//*[text()='Stats']")
                stats_tab.click()
                time.sleep(1.5)
            except Exception as fallback_exc:
                log.debug(
                    "Could not click Stats tab via fallback selector: %s", fallback_exc
                )

    def _assign_dates(self, driver: Any, matches: list[dict[str, Any]]) -> None:
        try:
            elements = driver.find_elements(
                By.XPATH, "//*[self::h3 or self::a[contains(@href, '/matches/')]]"
            )
            current_date = "N/A"
            match_idx = 0

            for element in elements:
                if element.tag_name == "h3":
                    date_text = element.text.strip()
                    # Intentionally local wall-clock time: resolves the site's own
                    # relative "today"/"tomorrow"/"yesterday" labels, not an
                    # absolute/comparable timestamp — UTC would be the wrong
                    # reference frame here.
                    if date_text.lower() == "today":
                        current_date = datetime.now().strftime("%A, %B %d, %Y")  # noqa: DTZ005
                    elif date_text.lower() == "tomorrow":
                        current_date = (
                            datetime.now() + timedelta(days=1)  # noqa: DTZ005
                        ).strftime("%A, %B %d, %Y")
                    elif date_text.lower() == "yesterday":
                        current_date = (
                            datetime.now() - timedelta(days=1)  # noqa: DTZ005
                        ).strftime("%A, %B %d, %Y")
                    else:
                        current_date = date_text
                elif element.tag_name == "a" and match_idx < len(matches):
                    matches[match_idx]["date"] = current_date
                    match_idx += 1
        except Exception as exc:
            log.debug("Error assigning dates: %s", exc)

    @staticmethod
    def _normalize_leagues(leagues: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(leagues, str):
            leagues = [leagues]

        normalized = tuple(
            dict.fromkeys(league.strip() for league in leagues if league.strip())
        )
        if not normalized:
            raise ValueError("At least one league is required.")

        unsupported = [l for l in normalized if l not in LEAGUE_CATALOG]
        if unsupported:
            raise ValueError(f"Unsupported leagues: {unsupported}")

        return normalized

    @staticmethod
    def _normalize_seasons(
        seasons: str | int | Iterable[str | int] | None,
    ) -> list[str]:
        if not seasons:
            return []

        values = [seasons] if isinstance(seasons, (str, int)) else list(seasons)
        return [str(v).strip() for v in values if str(v).strip()]

    @staticmethod
    def _get_current_season() -> str:
        now = datetime.now()  # noqa: DTZ005 — local calendar date, not a comparable timestamp
        if now.month >= 8:
            return f"{now.year}-{now.year + 1}"
        return f"{now.year - 1}-{now.year}"

    def _output_file(self, file_stem: str, season: str) -> Path:
        return self.output_dir / f"{file_stem}_match_stats_{season}.csv"
