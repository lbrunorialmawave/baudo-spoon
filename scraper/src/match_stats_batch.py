from __future__ import annotations

"""Batch fetching of FotMob match stats, replacing the legacy per-match
Selenium navigation (driver.get + click 'Stats' tab + scroll + DOM parse)
with parallel in-browser fetch() calls that read the same JSON payload
Next.js already embeds in every server-rendered match page.

Root cause of the old approach being slow: a full page load + hydration +
UI interaction (~6-7s) was paid for *every single match* to extract ~15
numbers that already exist as JSON inside the HTML response.

This module does N matches per round-trip:
  1. It reuses the already-authenticated/UC browser session (cookies,
     TLS fingerprint, Cloudflare clearance) — the same trick already used
     in stats_scraper._fetch_league_stats.
  2. It issues `fetch(url, {credentials:'include'})` for many match URLs
     concurrently from a single `execute_async_script` call, in small
     batches to stay polite with the origin.
  3. Parsing of __NEXT_DATA__ and extraction of the stats groups happens
     in JS, so only the small, already-shaped result crosses the
     Selenium wire — not the full HTML of every page.

No page rendering, no clicking, no scrolling, no fixed sleeps: what used
to take ~6-7s/match now costs roughly one HTTP round-trip, batched.
"""

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# Extracts __NEXT_DATA__ from raw match-page HTML and pulls out only the
# stats groups we care about, so the payload sent back to Python is tiny.
_JS_BATCH_FETCH_MATCH_STATS = r"""
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
        const data = JSON.parse(m[1]);
        const pp = (data.props || {}).pageProps || {};
        const content = pp.content || {};
        const statsRoot = ((content.stats || {}).stats) || [];

        if (!statsRoot.length) {
            // Page shell rendered without stats yet (e.g. very recent match).
            return { url, ok: false, error: 'empty stats (not finalised)' };
        }

        const groups = statsRoot.map(g => ({
            title: g.title || 'Top stats',
            stats: (g.stats || []).map(s => ({
                title: s.title || s.key || '',
                stats: s.stats || null,
            })).filter(s => s.title && s.stats),
        })).filter(g => g.stats.length);

        return { url, ok: true, groups };
    } catch (e) {
        return { url, ok: false, error: String(e) };
    }
}

Promise.all(urls.map(fetchOne)).then(done).catch(e => done(
    urls.map(u => ({ url: u, ok: false, error: String(e) }))
));
"""


def fetch_matches_batch(
    driver: Any,
    match_urls: list[str],
    batch_size: int = 12,
    inter_batch_delay: float = 0.4,
    script_timeout: int = 40,
) -> dict[str, list[dict[str, Any]] | None]:
    """Fetch match-stats groups for many matches in parallel batches.

    Args:
        driver: an already-warmed-up (driver.get(FOTMOB_BASE_URL) called)
            Selenium/UC driver — needed so fetch() carries valid cookies
            and passes Cloudflare's browser-fingerprint checks.
        match_urls: full match page URLs to fetch.
        batch_size: how many concurrent fetch() calls per round-trip.
            Kept modest (not 100+) to avoid looking like abuse to the
            origin; tune upward if you don't see failures/blocks.
        inter_batch_delay: pause between batches, seconds.
        script_timeout: seconds to wait for one batch's Promise.all to
            resolve (network-bound, not render-bound, so this is generous
            but rarely hit).

    Returns:
        {match_url: groups | None} — ``groups`` is the same
        section/stat-group shape produced by the old DOM scraper's
        ``stats_data`` (once fed through ``stats_from_next_data``), or
        ``None`` if that match could not be fetched/parsed.
    """
    results: dict[str, list[dict[str, Any]] | None] = {}
    if not match_urls:
        return results

    driver.set_script_timeout(script_timeout)
    total = len(match_urls)
    ok = 0
    failed = 0

    for start in range(0, total, batch_size):
        chunk = match_urls[start : start + batch_size]
        try:
            batch_result: list[dict[str, Any]] = driver.execute_async_script(
                _JS_BATCH_FETCH_MATCH_STATS, chunk
            )
        except Exception as exc:
            log.warning(
                "match_stats_batch: batch %d-%d failed entirely: %s",
                start,
                start + len(chunk),
                exc,
            )
            for url in chunk:
                results[url] = None
            failed += len(chunk)
            continue

        for item in batch_result or []:
            url = item.get("url")
            if item.get("ok"):
                results[url] = item.get("groups") or []
                ok += 1
            else:
                results[url] = None
                failed += 1
                log.debug(
                    "match_stats_batch: %s → %s",
                    url,
                    item.get("error") or item.get("status"),
                )

        log.info(
            "match_stats_batch: %d/%d done (ok=%d, failed=%d)",
            min(start + batch_size, total),
            total,
            ok,
            failed,
        )
        time.sleep(inter_batch_delay)

    return results


def stats_from_next_data(
    groups: list[dict[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    """Convert the JS-extracted stat groups into the same
    ``{section: {stat_name: [home, away]}}`` shape that
    ``parser.extract_stat_sections`` used to build from the DOM, so
    downstream code (row building, DB ingestion) needs no changes.
    """
    stats_data: dict[str, dict[str, list[Any]]] = {}
    for group in groups or []:
        section = group.get("title") or "Top stats"
        bucket = stats_data.setdefault(section, {})
        for stat in group.get("stats") or []:
            name = stat.get("title")
            values = stat.get("stats")
            if not name or not values or len(values) < 2:
                continue
            bucket[name] = values[:2]
    return stats_data
