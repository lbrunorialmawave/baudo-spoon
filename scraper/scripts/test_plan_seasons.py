"""End-to-end plan test: feed the real FotMob response into _plan_seasons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import LEAGUE_CATALOG  # noqa: E402
from src.stats_scraper import FotMobLeagueStatsScraper  # noqa: E402

RESP = Path(
    r"C:\Users\L.Brunori\Documents\Progetti\personal\analysis\response\response-data-leagues.json"
)


def main() -> None:
    payload = json.loads(RESP.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})

    meta = LEAGUE_CATALOG["Serie A"]
    scraper = FotMobLeagueStatsScraper(leagues=["Serie A"], seasons=["2025-2026"])
    plan = scraper._plan_seasons(stats, meta)  # noqa: SLF001
    print(f"Plan size: {len(plan)} seasons")
    for label, season_id, jobs in plan:
        url = jobs[0][2] if jobs else "(empty)"
        print(f"  {label} (id={season_id}): {len(jobs)} job(s) -> {url}")


if __name__ == "__main__":
    main()
