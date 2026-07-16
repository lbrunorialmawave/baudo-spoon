"""Validate the new topstats parser against the real CDN sample."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_scraper import _parse_topstats_payload  # noqa: E402

SAMPLE = Path(r"C:\Users\L.Brunori\Documents\Progetti\personal\analysis\response\topstats_27044.json")


def main() -> None:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    expanded = _parse_topstats_payload(payload, default_stat_type="players")
    print(f"Expanded {len(expanded)} tuples from {len(payload.get('TopLists', []))} raw TopLists")

    by_type: Counter[str] = Counter()
    for stype, name, rows in expanded:
        by_type[stype] += 1
    print(f"By stat_type: {dict(by_type)}")

    print("\nFirst 5 tuples:")
    for stype, name, rows in expanded[:5]:
        print(f"  {stype:<8s} {name:<40s} -> {len(rows)} rows")
        if rows:
            print(f"    sample: {rows[0]}")

    print("\nFirst 3 team tuples:")
    team_entries = [(s, n, r) for s, n, r in expanded if s == "teams"]
    for stype, name, rows in team_entries[:3]:
        print(f"  {stype:<8s} {name:<40s} -> {len(rows)} rows")
        if rows:
            print(f"    sample: {rows[0]}")

    # Check that team rows have team-shape (no ParticipantName; TeamId as entity)
    if team_entries:
        _, _, sample_team_rows = team_entries[0]
        print(f"\nTeam row keys: {sorted(sample_team_rows[0].keys())}")
    player_entries = [(s, n, r) for s, n, r in expanded if s == "players"]
    if player_entries:
        _, _, sample_player_rows = player_entries[0]
        print(f"Player row keys: {sorted(sample_player_rows[0].keys())}")


if __name__ == "__main__":
    main()
