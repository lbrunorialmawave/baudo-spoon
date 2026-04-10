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
from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG, SERIE_A, LeagueMeta
from .parser import (
    create_team_rows,
    extract_possession,
    extract_stat_sections,
    parse_match_link,
)

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
            for league_name in self.leagues:
                meta = LEAGUE_CATALOG[league_name]

                # If no season provided, we scrape the current one (e.g. 2023-2024 or 2024-2025).
                # For simplicity, if not provided we just use a default season string.
                # FotMob requires the exact season string like "2023-2024".
                seasons_to_scrape = self.seasons if self.seasons else [self._get_current_season()]

                for season in seasons_to_scrape:
                    log.info("Scraping league=%s season=%s", league_name, season)
                    all_results: list[dict[str, Any]] = []

                    for _round_num, round_results in self._scrape_season(driver, meta, season):
                        if round_results:
                            all_results.extend(round_results)
                            if on_round_complete is not None:
                                on_round_complete(league_name, season, pd.DataFrame(round_results))

                    if not all_results:
                        log.warning("No results found for %s %s", league_name, season)
                        continue

                    df = pd.DataFrame(all_results)
                    output_path = self._output_file(meta.file_stem, season)
                    df.to_csv(output_path, index=False, encoding="utf-8")
                    log.info("Wrote %d rows → %s", len(df), output_path)

                    outputs[(league_name, season)] = (df, output_path)

        return outputs

    def _scrape_season(
        self, driver: Any, meta: LeagueMeta, season: str
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        total_rounds = 38  # Defaulting to 38 for major leagues

        for round_num in range(1, total_rounds + 1):
            log.info("Starting Round %d/%d for %s", round_num, total_rounds, meta.display_name)

            try:
                round_results = self._get_matches_with_stats(driver, meta, season, round_num)
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
            log.debug("Scraping match %d/%d: %s vs %s", i, len(matches), match["home"], match["away"])
            home_row, away_row = create_team_rows(match, round_num)

            # Only scrape stats if match is finished
            if match["status"] in ("FT", "HT", "AET", "PEN"):
                stats = self._scrape_match_stats(driver, match["url"])
                
                # Flatten stats into rows
                for section, section_stats in stats.items():
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
                except Exception:
                    pass

            self._assign_dates(driver, matches)
            return matches

        except Exception as exc:
            log.warning("Error occurred fetching matches for round %d: %s", round_num, exc)
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
            stats_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Stats')]")))
            stats_tab.click()
            time.sleep(1.5)
        except Exception as exc:
            log.debug("Could not click Stats tab via wait: %s", exc)
            try:
                stats_tab = driver.find_element(By.XPATH, "//*[text()='Stats']")
                stats_tab.click()
                time.sleep(1.5)
            except Exception:
                pass

    def _assign_dates(self, driver: Any, matches: list[dict[str, Any]]) -> None:
        try:
            elements = driver.find_elements(By.XPATH, "//*[self::h3 or self::a[contains(@href, '/matches/')]]")
            current_date = "N/A"
            match_idx = 0

            for element in elements:
                if element.tag_name == "h3":
                    date_text = element.text.strip()
                    if date_text.lower() == "today":
                        current_date = datetime.now().strftime("%A, %B %d, %Y")
                    elif date_text.lower() == "tomorrow":
                        current_date = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d, %Y")
                    elif date_text.lower() == "yesterday":
                        current_date = (datetime.now() - timedelta(days=1)).strftime("%A, %B %d, %Y")
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
    def _normalize_seasons(seasons: str | int | Iterable[str | int] | None) -> list[str]:
        if not seasons:
            return []
        
        values = [seasons] if isinstance(seasons, (str, int)) else list(seasons)
        return [str(v).strip() for v in values if str(v).strip()]

    @staticmethod
    def _get_current_season() -> str:
        now = datetime.now()
        if now.month >= 8:
            return f"{now.year}-{now.year + 1}"
        return f"{now.year - 1}-{now.year}"

    def _output_file(self, file_stem: str, season: str) -> Path:
        return self.output_dir / f"{file_stem}_match_stats_{season}.csv"
