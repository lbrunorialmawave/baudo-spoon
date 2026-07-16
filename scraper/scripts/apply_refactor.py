"""Apply the 2-phase refactor to stats_scraper.py."""
from __future__ import annotations

from pathlib import Path

P = Path("C:/Users/L.Brunori/Documents/Progetti/personal/analysis/scraper/src/stats_scraper.py")
src = P.read_text(encoding="utf-8")

# ── 1. Sostituisci _fetch_one_stat (rimuovi ramo multi-TopList) ──────────────
old_fetch_one = '''async def _fetch_one_stat(
    client: httpx.AsyncClient,
    stat_type: str,
    category: str,
    url: str,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Fetch a single stat URL and return one or more ``(stat_type, stat_category, rows)`` tuples.

    For the multi-``TopList`` sentinel ``category == "top"`` the payload is
    expanded into one tuple per inner ``TopList``; otherwise the result is a
    single-element list.
    """
    try:
        resp = await client.get(url)
        if resp.status_code == 403:
            # Historic seasons lack advanced stats (xG, defensive_contributions, …).
            # A 403 here means the data never existed, not a rate-limit signal.
            log.debug("No data %s/%s (403): %s", stat_type, category, url)
            return []
        resp.raise_for_status()
        payload = resp.json()

        if category == "top" and isinstance(payload, dict) and payload.get("TopLists"):
            expanded = _parse_topstats_payload(payload, stat_type)
            log.debug("%s/top: %d inner TopLists expanded", stat_type, len(expanded))
            return expanded

        rows = _parse_stat_payload(payload, stat_type)
        log.debug("%s/%s: %d rows", stat_type, category, len(rows))
        return [(stat_type, category, rows)]
    except httpx.TimeoutException:
        log.warning("Timeout %s/%s [%s]", stat_type, category, url)
        return []
    except Exception as exc:
        log.warning("Fetch error %s/%s [%s]: %s", stat_type, category, url, exc)
        return []'''

new_fetch_one = '''async def _fetch_one_stat(
    client: httpx.AsyncClient,
    stat_type: str,
    category: str,
    url: str,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Fetch a single per-stat CDN file and return one ``(stat_type, category, rows)`` tuple.

    Individual stat files (e.g. ``goals.json``) contain the FULL ranking -- not
    a Top-3 preview. They always wrap a single ``TopLists[0]`` element which
    is consumed by ``_parse_stat_payload``.
    """
    try:
        resp = await client.get(url)
        if resp.status_code == 403:
            # Historic seasons lack advanced stats (xG, defensive_contributions, ...).
            # A 403 here means the data never existed, not a rate-limit signal.
            log.debug("No data %s/%s (403): %s", stat_type, category, url)
            return []
        resp.raise_for_status()
        payload = resp.json()
        rows = _parse_stat_payload(payload, stat_type)
        log.debug("%s/%s: %d rows", stat_type, category, len(rows))
        return [(stat_type, category, rows)]
    except httpx.TimeoutException:
        log.warning("Timeout %s/%s [%s]", stat_type, category, url)
        return []
    except Exception as exc:
        log.warning("Fetch error %s/%s [%s]: %s", stat_type, category, url, exc)
        return []


async def _discover_stat_urls(
    client: httpx.AsyncClient,
    topstats_url: str,
    default_stat_type: str = "players",
) -> list[tuple[str, str, str]]:
    """
    Phase 1: fetch the per-season ``topstats.json`` and expand its
    ``TopLists`` into one ``(stat_type, stat_name, StatLocation)`` job per
    inner ``TopList``.

    ``StatLocation`` is the absolute CDN URL to the full-ranking file (e.g.
    ``https://data.fotmob.com/stats/55/season/27044/goals.json``) which
    contains all players/teams, not just the Top 3 preview. Returns an
    empty list on any error.
    """
    try:
        resp = await client.get(topstats_url)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("Discover failed for %s: %s", topstats_url, exc)
        return []

    jobs: list[tuple[str, str, str]] = []
    top_lists: list[Any] = (payload.get("TopLists") or []) if isinstance(payload, dict) else []
    for top in top_lists:
        if not isinstance(top, dict):
            continue
        stat_name = top.get("StatName")
        stat_url = top.get("StatLocation")
        if not stat_name or not stat_url:
            continue
        entries = top.get("StatList") or []
        stat_type = _infer_stat_type(str(stat_name), entries, default_stat_type)
        jobs.append((stat_type, str(stat_name), str(stat_url)))
    return jobs


async def _fetch_full_season_stats(
    topstats_url: str,
    default_stat_type: str = "players",
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """
    Two-phase fetch for a single season:

      1. ``_discover_stat_urls`` reads ``topstats.json`` and extracts every
         ``StatLocation`` (the URL of the per-stat full ranking).
      2. All those URLs are fetched concurrently with ``_fetch_one_stat`` and
         their results flattened.

    Returns a list of ``(stat_type, stat_category, rows)`` -- one per
    successfully fetched stat file.
    """
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
        jobs = await _discover_stat_urls(client, topstats_url, default_stat_type)
        if not jobs:
            return []
        log.info(
            "Discovered %d stat URLs from %s",
            len(jobs),
            topstats_url.rsplit("/", 1)[-1],
        )
        nested = await asyncio.gather(
            *[_fetch_one_stat(client, stype, name, url) for stype, name, url in jobs],
            return_exceptions=False,
        )
    flat: list[tuple[str, str, list[dict[str, Any]]]] = []
    for item in nested:
        flat.extend(item)
    return flat'''

assert old_fetch_one in src, "old_fetch_one non trovato"
src = src.replace(old_fetch_one, new_fetch_one)
print("[1/4] _fetch_one_stat aggiornata, nuove funzioni aggiunte")

# ── 2. Rimuovi _parse_topstats_payload e _parse_stat_list (dead code) ────────
old_dead_block_marker_start = "def _parse_topstats_payload("
old_dead_block_marker_end_marker = "def _parse_stat_payload(payload: Any, stat_type: str) -> list[dict[str, Any]]:"  # noqa: E501
start_idx = src.index(old_dead_block_marker_start)
end_idx = src.index(old_dead_block_marker_end_marker)
# Taglia tutto da start_idx a end_idx (escluso)
removed = src[start_idx:end_idx]
print(f"[2/4] Rimuovo blocco dead-code ({len(removed)} caratteri) da offset {start_idx} a {end_idx}")
src = src[:start_idx] + src[end_idx:]

# ── 3. Modifica _plan_seasons: sentinel "__topstats__" + 1 job per stagione ─
old_plan = '''            url = f"{_CDN_BASE}/{rel_path}"
            jobs: list[tuple[str, str, str]] = [("players", "top", url)]
            plan.append((season_label, season_id, jobs))'''
new_plan = '''            url = f"{_CDN_BASE}/{rel_path}"
            # 1 discover-job per stagione: _fetch_full_season_stats espande
            # questo in N fetch paralleli (uno per ogni TopList di topstats.json).
            jobs: list[tuple[str, str, str]] = [("players", "__topstats__", url)]
            plan.append((season_label, season_id, jobs))'''
assert old_plan in src, "old_plan non trovato"
src = src.replace(old_plan, new_plan)
print("[3/4] _plan_seasons aggiornata con sentinel __topstats__")

# ── 4. Modifica _run_league per usare _fetch_full_season_stats ──────────────
old_run_league = '''            log.info(
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
                    )'''
new_run_league = '''            log.info(
                "[%s] %s (fotmob_id=%d): discover phase",
                league_name, season_label, fotmob_season_id,
            )
            # Il job "__topstats__" triggera una pipeline 2-fasi:
            # 1) discover degli StatLocation da topstats.json
            # 2) fetch parallelo di tutti i file di ranking completi
            _, _, topstats_url = jobs[0]
            results = asyncio.run(_fetch_full_season_stats(topstats_url))

            for stat_type, stat_category, rows in results:
                if rows:
                    log.info(
                        "[%s] %s | %s/%s: %d rows",
                        league_name, season_label, stat_type, stat_category, len(rows),
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
                        "[%s] %s | %s/%s: 0 rows",
                        league_name, season_label, stat_type, stat_category,
                    )'''
assert old_run_league in src, "old_run_league non trovato"
src = src.replace(old_run_league, new_run_league)
print("[4/4] _run_league aggiornata per pipeline 2-fasi")

P.write_text(src, encoding="utf-8")
print(f"\nFile aggiornato: {P} ({len(src)} chars)")
