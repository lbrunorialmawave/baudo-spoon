from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Generator
from typing import Any

import httpx

from .driver import get_managed_driver
from .models import FOTMOB_BASE_URL, LEAGUE_CATALOG, LeagueMeta

log = logging.getLogger(__name__)

_API_BASE = "https://www.fotmob.com"
_CDN_BASE = "https://data.fotmob.com"
_SEASON_ID_RE = re.compile(r"/season/(\d+)/")

# JavaScript that extracts ranked rows from a rendered FotMob stats page.
# arguments[0]: "players" | "teams"
_JS_EXTRACT_ROWS = r"""
const entityType = arguments[0];
const isPlayers = entityType === 'players';
const sel = isPlayers ? 'a[href*="/players/"]' : 'a[href*="/overview/"]';
const idRe = isPlayers ? /\/players\/(\d+)\// : /\/teams\/(\d+)\//;
const rows = [];
const seen = new Set();

function collectNums(el) {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    const out = [];
    let node;
    while ((node = walker.nextNode())) {
        const s = node.textContent.trim().replace(/,/g, '.');
        if (/^-?\d+(\.\d+)?$/.test(s)) out.push(parseFloat(s));
    }
    return out;
}

const root = document.querySelector('main') ||
             document.querySelector('[role="main"]') ||
             document.body;

for (const link of root.querySelectorAll(sel)) {
    const m = link.href.match(idRe);
    if (!m) continue;
    const eid = m[1];
    if (seen.has(eid)) continue;
    seen.add(eid);
    const name = link.textContent.trim();
    if (!name) continue;

    // Walk up to locate the row container (needs â‰¥ 3 child elements)
    let row = link.parentElement;
    for (let i = 0; i < 8 && row; i++) {
        if (row.childElementCount >= 3) break;
        row = row.parentElement;
    }
    if (!row) continue;

    let teamId = null, teamName = '';
    if (isPlayers) {
        const tl = row.querySelector('a[href*="/teams/"]');
        if (tl) {
            const tm = tl.href.match(/\/teams\/(\d+)\//);
            if (tm) teamId = tm[1];
            teamName = tl.textContent.trim();
        }
    }

    rows.push({ eid, name, teamId, teamName, nums: collectNums(row), href: isPlayers ? link.href : null });
}
return rows;
"""

# Tries several common __NEXT_DATA__ paths that Fotmob uses.
_JS_SEASON_INFO = """
try {
    const d = window.__NEXT_DATA__;
    if (!d) return null;
    const pp = (d.props || {}).pageProps || {};
    const league = pp.league || pp.data || {};
    const current = league.selectedSeason || league.season || {};
    const allSeasons = league.seasons || league.allSeasons || null;
    return {
        currentYear: current.year || league.year || null,
        currentId:   current.id   || null,
        allSeasons:  Array.isArray(allSeasons)
                     ? allSeasons.map(s => ({ id: s.id, year: s.year }))
                     : null,
    };
} catch(e) { return null; }
"""


def _norm_season(raw: str) -> str:
    """Normalise FotMob year format: '2024/2025' â†’ '2024-2025'."""
    return raw.strip().replace("/", "-")


def _infer_current_season() -> str:
    """Best-guess current football season from today's date."""
    today = datetime.date.today()
    year = today.year
    return f"{year}-{year + 1}" if today.month >= 7 else f"{year - 1}-{year}"


def _scroll_to_bottom(driver: Any, max_attempts: int = 30) -> None:
    """Trigger infinite-scroll until no new content loads."""
    prev_height = 0
    for _ in range(max_attempts):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)
        cur: int = driver.execute_script("return document.body.scrollHeight")
        if cur == prev_height:
            break
        prev_height = cur


def _parse_raw_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert JS-extracted raw rows into clean dicts with rank and value.

    Heuristic: first number that looks like a rank (positive int â‰¤ 1000) â†’ rank;
    last number â†’ stat value.
    """
    result: list[dict[str, Any]] = []
    for seq, d in enumerate(raw, 1):
        nums: list[float] = d.get("nums", [])
        rank: int = seq
        value: float | None = None

        if nums:
            first = nums[0]
            if first == int(first) and 1 <= int(first) <= 1000:
                rank = int(first)
                value = nums[-1] if len(nums) > 1 else None
            else:
                value = nums[-1]

        result.append(
            {
                "entity_id": int(d["eid"]),
                "entity_name": d["name"],
                "team_id": int(d["teamId"]) if d.get("teamId") else None,
                "team_name": d.get("teamName") or "",
                "rank": rank,
                "value": value,
                "entity_url": d.get("href") or None,
            }
        )
    return result


class _LegacyFotMobLeagueStatsScraper:
    """
    Scrape per-season player & team ranking stats from FotMob league stats pages.

    Workflow per league:
      1. Load the stats overview page (no season filter â†’ current season).
         Extract the fotmob_season_id from rendered category links and read
         available seasons from window.__NEXT_DATA__ / ?season= switcher links.
      2. For every requested season Ã— discovered category, navigate the dedicated
         ranking page and extract all entries via JavaScript DOM traversal.
    """

    def __init__(
        self,
        leagues: str | list[str],
        seasons: str | list[str] | None = None,
    ) -> None:
        self.leagues = [leagues] if isinstance(leagues, str) else list(leagues)
        if seasons is None:
            self.seasons: list[str] | None = None
        elif isinstance(seasons, str):
            self.seasons = [seasons]
        else:
            self.seasons = list(seasons)

    def run(
        self,
    ) -> Generator[tuple[str, str, int, str, str, list[dict[str, Any]]], None, None]:
        """
        Yields (league_name, season_label, fotmob_season_id, stat_type,
                stat_category, rows) for every scraped combination.
        """
        log.info(
            "Starting stats scrape | leagues=%s  seasons=%s",
            self.leagues,
            self.seasons or "[all]",
        )
        with get_managed_driver() as driver:
            driver.set_script_timeout(30)
            log.debug("Warming up browser session at %s", FOTMOB_BASE_URL)
            driver.get(FOTMOB_BASE_URL)
            time.sleep(2)
            log.debug("Browser session ready")

            for league_name in self.leagues:
                meta = LEAGUE_CATALOG[league_name]
                log.info("-- League: %s (comp_id=%s)", league_name, meta.comp_id)
                yield from self._run_league(driver, league_name, meta)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Per-league orchestration
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_league(
        self,
        driver: Any,
        league_name: str,
        meta: LeagueMeta,
    ) -> Generator[tuple[str, str, int, str, str, list[dict[str, Any]]], None, None]:
        stats_base = (
            f"{FOTMOB_BASE_URL}/it/leagues/{meta.comp_id}/stats/{meta.stats_slug}"
        )

        log.debug("Navigating to stats overview: %s", stats_base)
        season_map, categories, build_id = self._bootstrap(driver, meta, stats_base)
        if not season_map:
            log.warning("No seasons resolved for %s -- skipping", league_name)
            return

        log.info(
            "Plan: %d season(s) x %d categories = %d total requests | league=%s",
            len(season_map),
            len(categories),
            len(season_map) * len(categories),
            league_name,
        )

        for season_label, fotmob_season_id in season_map.items():
            log.info(
                "-- Season %s (fotmob_id=%d) | %d categories to scrape",
                season_label,
                fotmob_season_id,
                len(categories),
            )
            for idx, (stat_type, stat_category) in enumerate(categories, 1):
                log.debug(
                    "   [%d/%d] %s/%s", idx, len(categories), stat_type, stat_category
                )
                rows = self._scrape_category(
                    driver, meta, fotmob_season_id, stat_type, stat_category,
                    build_id=build_id,
                )
                if rows:
                    log.debug(
                        "   -> %d rows | %s/%s", len(rows), stat_type, stat_category
                    )
                    yield (
                        league_name,
                        season_label,
                        fotmob_season_id,
                        stat_type,
                        stat_category,
                        rows,
                    )
                else:
                    log.warning(
                        "   -> 0 rows | %s / %s / %s",
                        league_name,
                        stat_type,
                        stat_category,
                    )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Bootstrap: one navigation â†’ season map + categories
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _bootstrap(
        self,
        driver: Any,
        meta: LeagueMeta,
        stats_base: str,
    ) -> tuple[dict[str, int], list[tuple[str, str]], str | None]:
        """
        Navigate to the stats overview page (no season filter = current season).
        Returns:
            season_map  – {season_label: fotmob_season_id}, newest first.
            categories  – [(stat_type, stat_category), …]
            build_id    – Next.js build ID for _next/data API (may be None).
        """
        log.debug("Bootstrap: loading %s", stats_base)
        driver.get(stats_base)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        log.debug("Bootstrap: page body present, sleeping 2s for JS hydration")
        time.sleep(2)

        # â”€â”€ current season ID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        current_id = self._season_id_from_links(driver)
        if not current_id:
            log.error(
                "Bootstrap: no /stats/season/ links found on %s stats page -- "
                "page may not have rendered or bot-detection is active. URL: %s",
                meta.display_name,
                driver.current_url,
            )
            return {}, []
        log.info("Bootstrap: current fotmob_season_id=%d", current_id)

        # â”€â”€ categories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        categories = self._categories_from_links(driver, current_id)
        if not categories:
            n_defaults = len(_FALLBACK_PLAYER_CATEGORIES) + len(_FALLBACK_TEAM_CATEGORIES)
            log.warning(
                "Bootstrap: 0 category links for %s (fotmob_season_id=%d) -- "
                "using %d hardcoded defaults",
                meta.display_name, current_id, n_defaults,
            )
            categories = [
                *[("players", c) for c in _FALLBACK_PLAYER_CATEGORIES],
                *[("teams", c) for c in _FALLBACK_TEAM_CATEGORIES],
            ]
        else:
            log.info(
                "Bootstrap: discovered %d categories for %s: %s",
                len(categories),
                meta.display_name,
                [f"{t}/{c}" for t, c in categories],
            )

        # â”€â”€ season map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        season_map = self._build_season_map(driver, meta, stats_base, current_id)
        log.info(
            "Bootstrap: resolved seasons for %s -> %s",
            meta.display_name,
            list(season_map.keys()),
        )

        # Extract Next.js buildId for the JSON API strategy
        build_id: str | None = driver.execute_script(
            "try { return window.__NEXT_DATA__.buildId || null; } catch(e) { return null; }"
        )
        if build_id:
            log.info("Bootstrap: Next.js buildId=%s", build_id)
        else:
            log.warning("Bootstrap: could not extract __NEXT_DATA__.buildId -- JSON API unavailable")

        return season_map, categories, build_id

    def _build_season_map(
        self,
        driver: Any,
        meta: LeagueMeta,
        stats_base: str,
        current_id: int,
    ) -> dict[str, int]:
        """
        Build {season_label: fotmob_season_id} for all seasons to scrape.

        Priority order:
        1. If --seasons was specified, resolve only those labels.
        2. Otherwise try window.__NEXT_DATA__ for the full season list.
        3. Fall back to collecting ?season= links from the current page.
        4. Last resort: current season only, label inferred from today's date.
        """
        # Strategy 1: window.__NEXT_DATA__ (zero extra navigations)
        log.debug("Season map: querying __NEXT_DATA__")
        js_data = self._js_season_info(driver)
        current_label: str | None = None
        if js_data:
            raw_year = js_data.get("currentYear")
            if raw_year:
                current_label = _norm_season(str(raw_year))
                log.debug("Season map: current season from JS = %s", current_label)
            else:
                log.debug("Season map: __NEXT_DATA__ present but currentYear is null")
        else:
            log.debug("Season map: __NEXT_DATA__ not available")

        if self.seasons is not None:
            log.info(
                "Season map: --seasons filter active, resolving %d label(s): %s",
                len(self.seasons), self.seasons,
            )
            return self._resolve_requested(
                driver, meta, stats_base, current_id, current_label
            )

        # Build from __NEXT_DATA__ if it has the full list.
        all_seasons_js: list[dict[str, Any]] | None = (
            js_data.get("allSeasons") if js_data else None
        )
        # Strategy 2: full list from __NEXT_DATA__.allSeasons
        if all_seasons_js:
            result = {
                _norm_season(str(s["year"])): int(s["id"])
                for s in all_seasons_js
                if s.get("id") and s.get("year")
            }
            if result:
                log.info(
                    "Season map: %d seasons from __NEXT_DATA__ for %s: %s",
                    len(result), meta.display_name, sorted(result.keys(), reverse=True),
                )
                return dict(sorted(result.items(), reverse=True))
        else:
            log.debug("Season map: allSeasons absent from __NEXT_DATA__, trying switcher links")

        # Strategy 3: ?season= switcher links on the current page
        result = self._seasons_from_switcher(driver, stats_base, current_id, current_label)
        if result:
            log.info(
                "Season map: %d seasons from switcher links for %s: %s",
                len(result), meta.display_name, sorted(result.keys(), reverse=True),
            )
            return result

        # Strategy 4: last resort - current season only
        label = current_label or _infer_current_season()
        log.warning(
            "Season map: all strategies exhausted for %s -- "
            "falling back to current season only (%s)",
            meta.display_name, label,
        )
        return {label: current_id}

    def _resolve_requested(
        self,
        driver: Any,
        meta: LeagueMeta,
        stats_base: str,
        current_id: int,
        current_label: str | None,
    ) -> dict[str, int]:
        """Resolve only the season labels explicitly requested via --seasons."""
        assert self.seasons is not None
        result: dict[str, int] = {}
        for label in self.seasons:
            if label == current_label:
                log.debug("Resolve: %s matches current page (id=%d)", label, current_id)
                result[label] = current_id
                continue
            log.debug("Resolve: navigating to ?season=%s", label)
            sid = self._id_for_season_label(driver, stats_base, label)
            if sid:
                log.info("Resolve: %s -> fotmob_id=%d", label, sid)
                result[label] = sid
            else:
                log.warning(
                    "Resolve: could not find fotmob_id for %s / %s",
                    meta.display_name, label,
                )
        return result

    def _seasons_from_switcher(
        self,
        driver: Any,
        stats_base: str,
        current_id: int,
        current_label: str | None,
    ) -> dict[str, int]:
        """
        Collect ?season= links from the current page, navigate each one, and
        extract the corresponding fotmob_season_id.
        """
        result: dict[str, int] = {}
        if current_label:
            result[current_label] = current_id

        seen: set[str] = set()
        for el in driver.find_elements(By.CSS_SELECTOR, "a[href]"):
            href = el.get_attribute("href") or ""
            m = re.search(r"[?&]season=(\d{4}-\d{4})", href)
            if m:
                seen.add(m.group(1))

        log.debug("Switcher: found %d season label(s) in page links: %s", len(seen), sorted(seen))

        for label in sorted(seen, reverse=True):
            if label == current_label:
                result[label] = current_id
                continue
            log.debug("Switcher: resolving ?season=%s", label)
            sid = self._id_for_season_label(driver, stats_base, label)
            if sid:
                log.info("Switcher: %s -> fotmob_id=%d", label, sid)
                result[label] = sid
            else:
                log.warning("Switcher: could not resolve fotmob_id for season %s", label)

        return dict(sorted(result.items(), reverse=True))

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Helpers: DOM / JS extraction
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _season_id_from_links(driver: Any) -> int | None:
        """Return the first fotmob_season_id found in any stat category link."""
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/stats/season/']")
        log.debug("_season_id_from_links: %d candidate link(s)", len(links))
        for el in links:
            href = el.get_attribute("href") or ""
            m = _STAT_PATH_RE.search(href)
            if m:
                log.debug("_season_id_from_links: matched -> id=%s  href=%s", m.group(1), href)
                return int(m.group(1))
        log.debug("_season_id_from_links: no matching link found")
        return None

    @staticmethod
    def _categories_from_links(
        driver: Any, fotmob_season_id: int
    ) -> list[tuple[str, str]]:
        """Collect (stat_type, stat_category) pairs from rendered stat links."""
        seen: set[tuple[str, str]] = set()
        categories: list[tuple[str, str]] = []
        for el in driver.find_elements(By.CSS_SELECTOR, "a[href]"):
            href = el.get_attribute("href") or ""
            m = _STAT_PATH_RE.search(href)
            if m and int(m.group(1)) == fotmob_season_id:
                pair = (m.group(2), m.group(3))
                if pair not in seen:
                    seen.add(pair)
                    categories.append(pair)
        return categories

    @staticmethod
    def _js_season_info(driver: Any) -> dict[str, Any] | None:
        """Extract season info from window.__NEXT_DATA__ (Next.js SPA)."""
        try:
            result = driver.execute_script(_JS_SEASON_INFO)
            if isinstance(result, dict):
                log.debug(
                    "__NEXT_DATA__: currentYear=%s  currentId=%s  allSeasons count=%d",
                    result.get("currentYear"),
                    result.get("currentId"),
                    len(result.get("allSeasons") or []),
                )
                return result
            log.debug("__NEXT_DATA__: script returned %s (expected dict)", type(result).__name__)
        except Exception as exc:
            log.debug("__NEXT_DATA__: JS execution failed -- %s", exc)
        return None

    def _id_for_season_label(
        self, driver: Any, stats_base: str, season_label: str
    ) -> int | None:
        """Navigate to ?season=label and extract fotmob_season_id from stat links."""
        url = f"{stats_base}?season={season_label}"
        log.debug("_id_for_season_label: GET %s", url)
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            log.debug("_id_for_season_label: page loaded, sleeping 2s")
            time.sleep(2)
            sid = self._season_id_from_links(driver)
            if sid:
                log.debug("_id_for_season_label: %s -> id=%d", season_label, sid)
            else:
                log.warning("_id_for_season_label: no stat links on %s", url)
            return sid
        except Exception as exc:
            log.warning("_id_for_season_label: navigation failed for %s -- %s", season_label, exc)
            return None

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Row scraping
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _parse_stats_table(table: list[Any]) -> list[dict[str, Any]]:
        """Parse a raw JSON stats table (from __NEXT_DATA__ or _next/data) into row dicts."""
        rows: list[dict[str, Any]] = []
        for seq, entry in enumerate(table, 1):
            if not isinstance(entry, dict):
                continue
            entity_id = int(
                entry.get("pid") or entry.get("id") or entry.get("playerId") or 0
            )
            if not entity_id:
                continue
            team_id_raw = entry.get("tid") or entry.get("teamId")
            value_raw = entry.get("value") or entry.get("statValue") or entry.get("stat")
            rows.append({
                "entity_id": entity_id,
                "entity_name": str(
                    entry.get("pn") or entry.get("name") or entry.get("playerName") or ""
                ),
                "team_id": int(team_id_raw) if team_id_raw else None,
                "team_name": str(
                    entry.get("tn") or entry.get("teamName") or ""
                ),
                "rank": int(
                    entry.get("rankOrder") or entry.get("rank") or entry.get("pos") or seq
                ),
                "value": float(value_raw) if value_raw is not None else None,
            })
        return rows

    def _scrape_category(
        self,
        driver: Any,
        meta: LeagueMeta,
        fotmob_season_id: int,
        stat_type: str,
        stat_category: str,
        *,
        build_id: str | None = None,
    ) -> list[dict[str, Any]]:
        url = (
            f"{FOTMOB_BASE_URL}/it/leagues/{meta.comp_id}"
            f"/stats/season/{fotmob_season_id}"
            f"/{stat_type}/{stat_category}/{meta.stats_slug}"
        )
        log.debug("_scrape_category: GET %s", url)
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # ── Strategy 1: read __NEXT_DATA__ injected by SSR (fastest) ──────────
            ssr_rows = self._try_ssr_json(driver, stat_type, stat_category)
            if ssr_rows:
                log.debug(
                    "_scrape_category: %d rows from __NEXT_DATA__ SSR | %s/%s",
                    len(ssr_rows), stat_type, stat_category,
                )
                return ssr_rows

            # ── Strategy 2: fetch /_next/data/{buildId}/... JSON endpoint ─────────
            if build_id:
                api_rows = self._try_next_data_fetch(
                    driver, meta, build_id, fotmob_season_id, stat_type, stat_category
                )
                if api_rows:
                    log.debug(
                        "_scrape_category: %d rows from _next/data API | %s/%s",
                        len(api_rows), stat_type, stat_category,
                    )
                    return api_rows

            # ── Strategy 3: DOM rendering fallback ───────────────────────────────
            log.debug("_scrape_category: JSON strategies yielded nothing, trying DOM")
            time.sleep(1.5)
            try:
                WebDriverWait(driver, 8).until(
                    lambda d: d.execute_script(
                        r"return Array.from(document.querySelectorAll('a[href]'))"
                        r".some(a => /\/players\/\d+\//.test(a.href))"
                    )
                )
                log.debug("_scrape_category: player-profile links present")
            except Exception:
                log.warning(
                    "_scrape_category: DOM rendering timed out | url=%s", url
                )
            _scroll_to_bottom(driver)
            raw: list[dict[str, Any]] = driver.execute_script(
                _JS_EXTRACT_ROWS, stat_type
            )
            log.debug(
                "_scrape_category: JS returned %d raw row(s) | %s/%s",
                len(raw), stat_type, stat_category,
            )
            parsed = _parse_raw_rows(raw)
            if parsed:
                top = parsed[0]
                log.debug(
                    "_scrape_category: top entry -> #%d %s value=%s",
                    top["rank"], top["entity_name"], top["value"],
                )
            return parsed
        except Exception as exc:
            log.warning(
                "Scrape error – %s / %s / %s: %s",
                meta.display_name,
                stat_type,
                stat_category,
                exc,
            )
            return []

    # ── JSON extraction helpers ───────────────────────────────────────────────

    def _try_ssr_json(
        self, driver: Any, stat_type: str, stat_category: str
    ) -> list[dict[str, Any]]:
        """
        Read the ranking table from window.__NEXT_DATA__ (injected by SSR).
        Returns parsed rows, or empty list if not available.
        """
        try:
            result = driver.execute_script("""
                try {
                    const pp = window.__NEXT_DATA__.props.pageProps;
                    if (!pp) return {err: 'no pageProps'};
                    const ts = pp.topStats || pp.stats
                               || (pp.data && pp.data.topStats)
                               || (pp.data && pp.data.stats)
                               || pp.data || {};
                    const table = (typeof ts === 'object' && !Array.isArray(ts))
                        ? (ts.table || ts.rows || ts.entries || null)
                        : (Array.isArray(ts) ? ts : null);
                    if (!Array.isArray(table))
                        return {err: 'no table', ppKeys: Object.keys(pp).slice(0,15),
                                tsType: typeof ts,
                                tsKeys: (ts && typeof ts === 'object')
                                        ? Object.keys(ts).slice(0,10) : []};
                    return {ok: true, table: table};
                } catch(e) { return {err: String(e)}; }
            """)
        except Exception as exc:
            log.debug("_try_ssr_json: script error: %s", exc)
            return []

        if not result or result.get("err"):
            log.debug("_try_ssr_json: %s/%s -- %s", stat_type, stat_category, result)
            return []

        table: list[Any] = result.get("table", [])
        rows = self._parse_stats_table(table)
        if not rows and table:
            log.warning(
                "_try_ssr_json: table had %d entries but 0 valid rows; sample=%s",
                len(table), table[0] if table else "N/A",
            )
        return rows

    def _try_next_data_fetch(
        self,
        driver: Any,
        meta: LeagueMeta,
        build_id: str,
        fotmob_season_id: int,
        stat_type: str,
        stat_category: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch /_next/data/{buildId}/... from within the browser so that
        session cookies and headers are preserved.
        """
        data_url = (
            f"/_next/data/{build_id}/en/leagues/{meta.comp_id}"
            f"/stats/season/{fotmob_season_id}/{stat_type}/{stat_category}/{meta.stats_slug}.json"
            f"?lng=en&id={meta.comp_id}&season={fotmob_season_id}"
            f"&type={stat_type}&stat={stat_category}&slug={meta.stats_slug}"
        )
        log.debug("_try_next_data_fetch: GET %s", data_url)
        try:
            result = driver.execute_async_script(
                """
                const url = arguments[0], cb = arguments[1];
                fetch(url, {credentials: 'include'})
                    .then(r => r.ok ? r.json() : Promise.reject('HTTP ' + r.status))
                    .then(data => cb({ok: true, data}))
                    .catch(e => cb({ok: false, err: String(e)}));
                """,
                data_url,
            )
        except Exception as exc:
            log.debug("_try_next_data_fetch: script error: %s", exc)
            return []

        if not result or not result.get("ok"):
            log.warning("_try_next_data_fetch: fetch failed: %s", result)
            return []

        page_props = (result.get("data") or {}).get("pageProps") or {}
        ts = (
            page_props.get("topStats")
            or page_props.get("stats")
            or (page_props.get("data") or {}).get("topStats")
            or (page_props.get("data") or {}).get("stats")
            or page_props.get("data")
            or {}
        )
        if isinstance(ts, dict):
            table = ts.get("table") or ts.get("rows") or ts.get("entries") or []
        elif isinstance(ts, list):
            table = ts
        else:
            table = []

        if not table:
            log.warning(
                "_try_next_data_fetch: no table in response; pageProps keys=%s",
                sorted(page_props.keys())[:15],
            )
            return []

        rows = self._parse_stats_table(table)
        if not rows:
            log.warning(
                "_try_next_data_fetch: table had %d entries but 0 valid rows; sample=%s",
                len(table), table[0] if table else "N/A",
            )
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# Module-level async helpers
# ──────────────────────────────────────────────────────────────────────────────


async def _fetch_all_stats(
    jobs: list[tuple[str, str, str]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Concurrently fetch all stat-category URLs and return parsed rows."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
    ) as client:
        return list(
            await asyncio.gather(
                *[_fetch_one_stat(client, stype, name, url) for stype, name, url in jobs],
                return_exceptions=False,
            )
        )


async def _fetch_one_stat(
    client: httpx.AsyncClient,
    stat_type: str,
    category: str,
    url: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    try:
        resp = await client.get(url)
        if resp.status_code == 403:
            # Historic seasons lack advanced stats (xG, defensive_contributions, …).
            # A 403 here means the data never existed, not a rate-limit signal.
            log.debug("No data %s/%s (403): %s", stat_type, category, url)
            return (stat_type, category, [])
        resp.raise_for_status()
        rows = _parse_stat_payload(resp.json(), stat_type)
        log.debug("%s/%s: %d rows", stat_type, category, len(rows))
        return (stat_type, category, rows)
    except httpx.TimeoutException:
        log.warning("Timeout %s/%s [%s]", stat_type, category, url)
        return (stat_type, category, [])
    except Exception as exc:
        log.warning("Fetch error %s/%s [%s]: %s", stat_type, category, url, exc)
        return (stat_type, category, [])


def _parse_stat_payload(payload: Any, stat_type: str) -> list[dict[str, Any]]:
    """
    Parse a ``data.fotmob.com`` stat JSON response.

    Real CDN shape (note FotMob's own typo ``ParticiantId``)::

        {
          "TopLists": [{
            "StatList": [
              {
                "ParticiantId": 39540,   # typo in FotMob API – no 'p' after 'Partici'
                "ParticipantName": "...",
                "TeamId": 9882,
                "TeamName": "...",
                "Rank": 1,
                "StatValue": 26.0
              }
            ]
          }],
          "LeagueName": "..."
        }

    For team stats ``ParticiantId`` is always 0; ``TeamId`` is the entity key.
    """
    if not isinstance(payload, dict):
        return []
    top_lists: list[Any] = payload.get("TopLists", [])
    if not top_lists:
        return []

    entries: list[Any] = top_lists[0].get("StatList", [])
    rows: list[dict[str, Any]] = []

    for entry in entries:
        # FotMob's intentional(?) API typo: "ParticiantId" (missing 'p')
        participant_id = entry.get("ParticiantId") or 0
        team_id_raw = entry.get("TeamId")
        value_raw = entry.get("StatValue")
        rank_raw = entry.get("Rank")

        if stat_type == "players":
            entity_id = int(participant_id)
            if not entity_id:
                continue
            row: dict[str, Any] = {
                "entity_id": entity_id,
                "entity_name": str(entry.get("ParticipantName") or ""),
                "rank": int(rank_raw) if rank_raw is not None else None,
                "value": float(value_raw) if value_raw is not None else None,
                "team_id": int(team_id_raw) if team_id_raw else None,
                "team_name": str(entry.get("TeamName") or ""),
            }
        else:
            # For team stats the real entity is TeamId; ParticiantId is always 0
            entity_id = int(team_id_raw) if team_id_raw else 0
            if not entity_id:
                continue
            row = {
                "entity_id": entity_id,
                "entity_name": str(entry.get("ParticipantName") or ""),
                "rank": int(rank_raw) if rank_raw is not None else None,
                "value": float(value_raw) if value_raw is not None else None,
            }

        rows.append(row)

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Public scraper
# ──────────────────────────────────────────────────────────────────────────────


class FotMobLeagueStatsScraper:
    """
    API-based scraper for FotMob per-season player and team ranking stats.

    One credentialed fetch via the browser retrieves the league metadata
    (including stat-category URLs for the current season and season IDs for
    all historical seasons).  All stat-category data is then fetched
    concurrently via httpx from FotMob's public CDN, keeping the Selenium
    session to the strict minimum.
    """

    def __init__(
        self,
        leagues: str | list[str],
        seasons: str | list[str] | None = None,
    ) -> None:
        self.leagues = [leagues] if isinstance(leagues, str) else list(leagues)
        if seasons is None:
            self.seasons: list[str] | None = None
        elif isinstance(seasons, str):
            self.seasons = [seasons]
        else:
            self.seasons = list(seasons)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
    ) -> Generator[tuple[str, str, int, str, str, list[dict[str, Any]]], None, None]:
        """
        Yields ``(league_name, season_label, fotmob_season_id, stat_type,
        stat_category, rows)`` for every scraped combination.
        """
        log.info(
            "Stats scrape starting | leagues=%s  seasons=%s",
            self.leagues,
            self.seasons or "[all]",
        )
        with get_managed_driver() as driver:
            driver.set_script_timeout(60)
            driver.get(FOTMOB_BASE_URL)
            time.sleep(2)

            for league_name in self.leagues:
                meta = LEAGUE_CATALOG[league_name]
                log.info("League: %s (comp_id=%s)", league_name, meta.comp_id)
                yield from self._run_league(driver, meta, league_name)

    # ------------------------------------------------------------------
    # Per-league orchestration
    # ------------------------------------------------------------------

    def _run_league(
        self,
        driver: Any,
        meta: LeagueMeta,
        league_name: str,
    ) -> Generator[tuple[str, str, int, str, str, list[dict[str, Any]]], None, None]:
        raw = self._fetch_league_stats(driver, meta)
        season_plan = self._plan_seasons(raw, meta)

        wanted: set[str] | None = (
            {_norm_season(s) for s in self.seasons} if self.seasons else None
        )

        for season_label, fotmob_season_id, jobs in season_plan:
            if wanted is not None and season_label not in wanted:
                continue

            log.info(
                "[%s] %s (fotmob_id=%d): %d stat jobs",
                league_name, season_label, fotmob_season_id, len(jobs),
            )
            results = asyncio.run(_fetch_all_stats(jobs))

            for stat_type, stat_category, rows in results:
                if rows:
                    yield (
                        league_name,
                        season_label,
                        fotmob_season_id,
                        stat_type,
                        stat_category,
                        rows,
                    )
                else:
                    log.warning(
                        "[%s] %s | %s/%s: 0 rows",
                        league_name, season_label, stat_type, stat_category,
                    )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_league_stats(self, driver: Any, meta: LeagueMeta) -> dict[str, Any]:
        """
        Execute a credentialed ``fetch()`` from within the browser to retrieve
        the FotMob league API payload, returning only the minimal fields needed
        for stat planning (avoids serialising the full multi-MB response).
        """
        url = f"{_API_BASE}/api/data/leagues?id={meta.comp_id}&ccode3={meta.country_code}"
        log.debug("Browser fetch: %s", url)

        result: dict[str, Any] = driver.execute_async_script(
            """
            const [url, done] = arguments;
            fetch(url, {credentials: 'include'})
                .then(r => {
                    if (!r.ok) { done({ok: false, status: r.status}); return; }
                    return r.json();
                })
                .then(d => {
                    if (!d) return;
                    const s = d.stats || {};
                    done({
                        ok: true,
                        players: (s.players || []).map(e => ({
                            name: e.name,
                            fetchAllUrl: e.fetchAllUrl,
                        })),
                        teams: (s.teams || []).map(e => ({
                            name: e.name,
                            fetchAllUrl: e.fetchAllUrl,
                        })),
                        seasonStatLinks: (s.seasonStatLinks || []).map(e => ({
                            Name: e.Name,
                            TournamentId: e.TournamentId,
                        })),
                    });
                })
                .catch(e => done({ok: false, error: String(e)}));
            """,
            url,
        )

        if not result or not result.get("ok"):
            raise RuntimeError(
                f"League stats fetch failed for {meta.display_name}: {result}"
            )
        return result

    # ------------------------------------------------------------------
    # Season / category planning
    # ------------------------------------------------------------------

    def _plan_seasons(
        self,
        stats: dict[str, Any],
        meta: LeagueMeta,
    ) -> list[tuple[str, int, list[tuple[str, str, str]]]]:
        """
        Build the scraping plan as a list of
        ``(season_label, fotmob_season_id, jobs)`` triples where each job is
        ``(stat_type, stat_name, fetch_url)``.

        The current season's jobs are taken verbatim from the API response.
        Historic seasons reuse the same stat names with the season ID swapped
        in the CDN URL path.
        """
        players: list[dict[str, Any]] = stats.get("players", [])
        teams: list[dict[str, Any]] = stats.get("teams", [])
        season_links: list[dict[str, Any]] = stats.get("seasonStatLinks", [])

        player_catalog: list[tuple[str, str, str]] = [
            ("players", e["name"], e["fetchAllUrl"])
            for e in players
            if e.get("name") and e.get("fetchAllUrl")
        ]
        team_catalog: list[tuple[str, str, str]] = [
            ("teams", e["name"], e["fetchAllUrl"])
            for e in teams
            if e.get("name") and e.get("fetchAllUrl")
        ]
        full_catalog = [*player_catalog, *team_catalog]

        current_season_id: int | None = None
        if player_catalog:
            m = _SEASON_ID_RE.search(player_catalog[0][2])
            if m:
                current_season_id = int(m.group(1))

        plan: list[tuple[str, int, list[tuple[str, str, str]]]] = []

        for link in season_links:
            season_id = int(link["TournamentId"])
            season_label = _norm_season(link["Name"])

            if season_id == current_season_id:
                jobs: list[tuple[str, str, str]] = list(full_catalog)
            else:
                jobs = [
                    (stype, name, _SEASON_ID_RE.sub(f"/season/{season_id}/", url))
                    for stype, name, url in full_catalog
                ]

            plan.append((season_label, season_id, jobs))

        return plan

