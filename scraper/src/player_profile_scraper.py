from __future__ import annotations

"""Fetch player role/position via browser page navigation.

FotMob's /api/data/playerData endpoint requires an x-mas HMAC token
computed per-request (path + timestamp) — it cannot be reused across
player IDs or copied from DevTools. This module avoids the API entirely:
it reads positionDescription from window.__NEXT_DATA__, embedded as
inline JSON in every server-rendered Next.js page — but instead of a full
page navigation per player (driver.get + hydration wait), it batches many
players per round-trip via in-browser fetch() of the raw HTML, the same
technique used in match_stats_batch.py and stats_scraper._fetch_league_stats.
"""

import logging
import re
import time
import unicodedata
from typing import Any

from .driver import get_managed_driver
from .models import FOTMOB_BASE_URL
from .roles_bridge import extract_profile_from_player_data

log = logging.getLogger(__name__)

_BATCH_SIZE = 15            # concurrent fetch() calls per round-trip
_INTER_BATCH_DELAY = 0.4    # seconds between batches
_SCRIPT_TIMEOUT = 40        # seconds, network-bound not render-bound


def _slugify(name: str) -> str:
    """Convert a player name to a FotMob URL slug.

    Examples:
        "Donyell Malen"       -> "donyell-malen"
        "Anastasios Douvikas" -> "anastasios-douvikas"
        "Andréa Le Borgne"    -> "andrea-le-borgne"
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _player_url(player_id: int, player_name: str, url_map: dict[int, str]) -> str:
    """Return the best available player page URL."""
    if player_id in url_map:
        return url_map[player_id]
    slug = _slugify(player_name)
    return f"https://www.fotmob.com/players/{player_id}/overview/{slug}"


# Fetches many player pages concurrently, extracts __NEXT_DATA__ from the
# raw HTML text (no rendering needed) and pulls out positionDescription
# using the same fallback path chain the old per-page script used.
_JS_BATCH_FETCH_POSITIONS = r"""
const [urls, done] = arguments;

async function fetchOne(url) {
    try {
        const res = await fetch(url, { credentials: 'include' });
        if (!res.ok) {
            return { url, ok: false, status: res.status };
        }
        const html = await res.text();
        const m = html.match(
            /<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/
        );
        if (!m) {
            return { url, ok: false, error: 'no __NEXT_DATA__' };
        }
        const d = JSON.parse(m[1]);
        const pp = (d.props || {}).pageProps || {};
        const profile =
            pp.playerProfile
            || pp.player
            || pp.profileData
            || pp.data
            || (pp.initialProps || {}).playerProfile
            || null;
        if (!profile) {
            return { url, ok: false, error: 'no profile key' };
        }
        const pos = profile.positionDescription || null;
        if (!pos) {
            return { url, ok: false, error: 'no positionDescription' };
        }
        return { url, ok: true, positionDescription: pos };
    } catch (e) {
        return { url, ok: false, error: String(e) };
    }
}

Promise.all(urls.map(fetchOne)).then(done).catch(e => done(
    urls.map(u => ({ url: u, ok: false, error: String(e) }))
));
"""


def _fetch_positions_batch(
    driver: Any,
    urls: list[str],
    batch_size: int = _BATCH_SIZE,
    inter_batch_delay: float = _INTER_BATCH_DELAY,
) -> dict[str, dict[str, Any] | None]:
    """Fetch positionDescription for many player URLs in parallel batches.

    Returns {url: positionDescription | None}.
    """
    results: dict[str, dict[str, Any] | None] = {}
    if not urls:
        return results

    driver.set_script_timeout(_SCRIPT_TIMEOUT)
    total = len(urls)
    ok = 0
    errors = 0

    for start in range(0, total, batch_size):
        chunk = urls[start : start + batch_size]
        try:
            batch_result: list[dict[str, Any]] = driver.execute_async_script(
                _JS_BATCH_FETCH_POSITIONS, chunk
            )
        except Exception as exc:
            log.warning(
                "player_profile_batch: batch %d-%d failed entirely: %s",
                start, start + len(chunk), exc,
            )
            for url in chunk:
                results[url] = None
            errors += len(chunk)
            continue

        for item in batch_result or []:
            url = item.get("url")
            if item.get("ok"):
                results[url] = item.get("positionDescription")
                ok += 1
            else:
                results[url] = None
                errors += 1
                log.debug("player_profile_batch: %s → %s", url, item.get("error") or item.get("status"))

        log.info(
            "player_profile: %d/%d done (ok=%d, errors=%d)",
            min(start + batch_size, total), total, ok, errors,
        )
        time.sleep(inter_batch_delay)

    return results


def fetch_player_profiles(
    player_ids: dict[int, str],
    player_url_map: dict[int, str] | None = None,
    batch_log_interval: int = 50,
) -> list[dict[str, Any]]:
    """Fetch positionDescription for every player and build profile dicts.

    Args:
        player_ids: {player_fotmob_id: player_name}
        player_url_map: {player_fotmob_id: full_page_url} collected during stats scrape.
            Players not in this map fall back to the generic /players/{id} URL.
        batch_log_interval: unused, kept for call-site compatibility.

    Returns:
        List of player_profiles-compatible dicts ready for upsert_player_profiles().
    """
    url_map = player_url_map or {}
    total = len(player_ids)

    # Build the full url list up front so we can batch-fetch it in one pass
    # instead of navigating to each player's page sequentially.
    urls_by_player: dict[int, str] = {
        player_id: _player_url(player_id, name, url_map)
        for player_id, name in player_ids.items()
    }

    with get_managed_driver() as driver:
        log.debug("Warming browser session at %s", FOTMOB_BASE_URL)
        driver.get(FOTMOB_BASE_URL)
        time.sleep(2)
        log.debug("Browser ready — starting batched position fetch (%d players)", total)

        position_by_url = _fetch_positions_batch(driver, list(urls_by_player.values()))

    profiles: list[dict[str, Any]] = []
    ok = 0
    errors = 0
    for player_id, name in player_ids.items():
        url = urls_by_player[player_id]
        pos = position_by_url.get(url)
        if pos is None:
            errors += 1
            continue
        profiles.append(
            extract_profile_from_player_data(player_id, name, {"positionDescription": pos})
        )
        ok += 1

    log.info("player_profile: finished — %d/%d ok, %d errors", ok, total, errors)
    return profiles