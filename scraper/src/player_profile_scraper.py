from __future__ import annotations

"""Fetch player role/position via browser page navigation.

FotMob's /api/data/playerData endpoint requires an x-mas HMAC token
computed per-request (path + timestamp) — it cannot be reused across
player IDs or copied from DevTools. This module avoids the API entirely:
it navigates to each player's public page and reads positionDescription
from window.__NEXT_DATA__, embedded as inline JSON in every server-rendered
Next.js page.
"""

import json
import logging
import re
import time
import unicodedata
from typing import Any

from .driver import get_managed_driver
from .models import FOTMOB_BASE_URL
from .roles_bridge import extract_profile_from_player_data

log = logging.getLogger(__name__)

_INTER_REQUEST_DELAY = 0.5   # seconds between page navigations
_HYDRATION_WAIT = 1.2        # seconds after driver.get() for Next.js hydration


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


# Tries multiple __NEXT_DATA__ paths used by different FotMob page versions.
# Returns a JSON object {ok, positionDescription} or {error, ...debug keys}.
_JS_EXTRACT_POSITION = """
try {
    const d = window.__NEXT_DATA__;
    if (!d) return JSON.stringify({error: 'no __NEXT_DATA__'});
    const pp = ((d.props || {}).pageProps) || {};
    const profile =
        pp.playerProfile
        || pp.player
        || pp.profileData
        || pp.data
        || (pp.initialProps || {}).playerProfile
        || null;
    if (!profile) {
        return JSON.stringify({error: 'no profile key', ppKeys: Object.keys(pp).slice(0, 20)});
    }
    const pos = profile.positionDescription || null;
    if (!pos) {
        return JSON.stringify({error: 'no positionDescription', profileKeys: Object.keys(profile).slice(0, 20)});
    }
    return JSON.stringify({ok: true, positionDescription: pos});
} catch(e) { return JSON.stringify({error: String(e)}); }
"""


def fetch_player_profiles(
    player_ids: dict[int, str],
    player_url_map: dict[int, str] | None = None,
    batch_log_interval: int = 50,
) -> list[dict[str, Any]]:
    """Navigate to each player's FotMob page and extract positionDescription.

    Args:
        player_ids: {player_fotmob_id: player_name}
        player_url_map: {player_fotmob_id: full_page_url} collected during stats scrape.
            Players not in this map fall back to the generic /players/{id} URL.
        batch_log_interval: Log progress every N players.

    Returns:
        List of player_profiles-compatible dicts ready for upsert_player_profiles().
    """
    url_map = player_url_map or {}
    profiles: list[dict[str, Any]] = []
    total = len(player_ids)
    ok = 0
    errors = 0

    with get_managed_driver() as driver:
        driver.set_script_timeout(20)

        log.debug("Warming browser session at %s", FOTMOB_BASE_URL)
        driver.get(FOTMOB_BASE_URL)
        time.sleep(2)
        log.debug("Browser ready for player profile navigation")

        for idx, (player_id, player_name) in enumerate(player_ids.items(), 1):
            url = _player_url(player_id, player_name, url_map)
            try:
                driver.get(url)
                time.sleep(_HYDRATION_WAIT)

                raw: str | None = driver.execute_script(_JS_EXTRACT_POSITION)
                if raw is None:
                    log.warning(
                        "player_profile: script returned None for player_id=%d (%s)",
                        player_id, player_name,
                    )
                    errors += 1
                else:
                    result = json.loads(raw)
                    if result.get("ok"):
                        profile = extract_profile_from_player_data(
                            player_id,
                            player_name,
                            {"positionDescription": result["positionDescription"]},
                        )
                        profiles.append(profile)
                        ok += 1
                    else:
                        log.warning(
                            "player_profile: extraction failed player_id=%d (%s): %s",
                            player_id, player_name, result,
                        )
                        errors += 1
            except Exception as exc:
                log.warning(
                    "player_profile: error for player_id=%d (%s): %s",
                    player_id, player_name, exc,
                )
                errors += 1

            if idx % batch_log_interval == 0:
                log.info(
                    "player_profile: %d/%d done (ok=%d, errors=%d)",
                    idx, total, ok, errors,
                )

            time.sleep(_INTER_REQUEST_DELAY)

    log.info(
        "player_profile: finished — %d/%d ok, %d errors",
        ok, total, errors,
    )
    return profiles
