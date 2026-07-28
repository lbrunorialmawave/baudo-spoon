"""Test the FotMob suggest/search API against our current matching strategy.

Compares results from:
1. The public FotMob suggest API (no auth required)
2. Our current DB-based matching strategy

Usage:
    python scripts/test_fotmob_suggest.py [--player "Lapo Nava"]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
import urllib.parse
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_fotmob_suggest")

FOTMOB_SUGGEST_URL = (
    "https://www.fotmob.com/api/data/search/suggest"
    "?hits=50&lang=it%2Cen%2Cfr&term={}"
)


def fotmob_suggest(term: str) -> list[dict]:
    """Call the FotMob suggest API and return player suggestions."""
    url = FOTMOB_SUGGEST_URL.format(urllib.parse.quote(term))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    players: list[dict] = []
    for group in data:
        for s in group.get("suggestions", []):
            if s.get("type") == "player":
                players.append({
                    "id": int(s["id"]),
                    "name": s["name"],
                    "team_id": s.get("teamId"),
                    "team_name": s.get("teamName"),
                    "score": s.get("score", 0),
                })
    return players


def search_via_db(
    player_name: str,
    team_name: str | None = None,
) -> list[dict]:
    """Simulate our current matching strategy (pure Python, no DB needed)."""
    from ml.data.import_quotations import (
        normalise_player_name,
        normalise_team,
        apply_team_alias,
        last_name_token,
    )

    name_norm = normalise_player_name(player_name)
    surname = last_name_token(name_norm)

    results: list[dict] = [{"source": "simulated", "surname": surname}]
    if team_name:
        team_norm = normalise_team(team_name)
        team_canon = apply_team_alias(team_norm)
        results[0]["team_norm"] = team_norm
        results[0]["team_canonical"] = team_canon

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Test FotMob suggest API")
    parser.add_argument(
        "--player", "-p",
        default="Lapo Nava",
        help="Player name to search (default: Lapo Nava)",
    )
    parser.add_argument(
        "--team", "-t",
        default=None,
        help="Team name (optional, for normalisation comparison)",
    )
    args = parser.parse_args()

    # ── 1. FotMob Suggest API ─────────────────────────────────
    log.info("🔍 Searching FotMob suggest API for: %s", args.player)
    try:
        suggestions = fotmob_suggest(args.player)
    except Exception as e:
        log.error("❌ FotMob suggest API failed: %s", e)
        suggestions = []

    if suggestions:
        log.info("✅ Found %d result(s):", len(suggestions))
        for s in suggestions:
            print(
                f"   ID={s['id']:>8}  score={s['score']:>7}  "
                f"{s['name']:30s}  [{s['team_name']}]"
            )
    else:
        log.warning("⚠️  No results from FotMob suggest API")

    # ── 2. Test with surname only ────────────────────────────
    print()
    log.info("🔍 Searching by surname only…")
    surname = args.player.split()[-1] if " " in args.player else args.player
    try:
        surname_results = fotmob_suggest(surname)
    except Exception as e:
        log.error("❌ FotMob suggest API failed: %s", e)
        surname_results = []

    if surname_results:
        log.info("✅ Found %d result(s) for surname '%s':", len(surname_results), surname)
        for s in surname_results:
            match = (
                " <<< MATCH" if s["name"].lower() == args.player.lower() else ""
            )
            print(
                f"   ID={s['id']:>8}  score={s['score']:>7}  "
                f"{s['name']:30s}  [{s['team_name']}]{match}"
            )
    else:
        log.warning("⚠️  No results for surname '%s'", surname)

    # ── 3. Normalisation comparison ───────────────────────────
    if args.team:
        from ml.data.import_quotations import (
            normalise_player_name,
            normalise_team,
            apply_team_alias,
            last_name_token,
        )

        print()
        log.info("📐 Normalisation analysis:")
        print(f"   Original name : {args.player}")
        print(f"   Normalised    : {normalise_player_name(args.player)}")
        print(f"   Surname token : {last_name_token(normalise_player_name(args.player))}")
        print(f"   Original team : {args.team}")
        print(f"   Normalised    : {normalise_team(args.team)}")
        print(f"   Canonical     : {apply_team_alias(normalise_team(args.team))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
