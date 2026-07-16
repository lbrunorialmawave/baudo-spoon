"""Aumenta read=20.0 → read=60.0 SOLO in _fetch_full_season_stats."""
from pathlib import Path

p = Path(r"c:\Users\L.Brunori\Documents\Progetti\personal\analysis\scraper\src\stats_scraper.py")
src = p.read_text(encoding="utf-8")

# Indentazione reale: 8 spazi per 'timeout=', 12 spazi per 'jobs = await...'
ANCHOR_OLD = (
    "        timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),\n"
    "        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),\n"
    "        follow_redirects=True,\n"
    "    ) as client:\n"
    "        jobs = await _discover_stat_urls(client, topstats_url, default_stat_type)"
)

ANCHOR_NEW = (
    "        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),\n"
    "        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),\n"
    "        follow_redirects=True,\n"
    "    ) as client:\n"
    "        jobs = await _discover_stat_urls(client, topstats_url, default_stat_type)"
)

count = src.count(ANCHOR_OLD)
assert count == 1, f"Atteso 1 match univoco in _fetch_full_season_stats, trovati {count}"

new = src.replace(ANCHOR_OLD, ANCHOR_NEW)
p.write_text(new, encoding="utf-8")
print("[OK] Timeout aggiornato: connect=10.0s, read=60.0s, write=10.0s, pool=10.0s")
print("     Modificata SOLO _fetch_full_season_stats (la legacy _fetch_all_stats resta intatta)")
