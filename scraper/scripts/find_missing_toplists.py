"""Find why 2 TopLists produce 0 rows after parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_scraper import _parse_topstats_payload  # noqa: E402

SAMPLE = Path(r"C:\Users\L.Brunori\Documents\Progetti\personal\analysis\response\topstats_27044.json")


def main() -> None:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    raw = payload.get("TopLists", [])
    expanded = _parse_topstats_payload(payload, default_stat_type="players")
    expanded_names = {n for _, n, _ in expanded}

    missing = [tl for tl in raw if tl.get("StatName") not in expanded_names]
    print(f"TopLists raw={len(raw)} parsed={len(expanded)} missing={len(missing)}")
    for tl in missing:
        name = tl.get("StatName")
        rows = tl.get("StatList") or []
        print(f"\nMissing TopList: StatName={name!r} rows={len(rows)}")
        if rows:
            print(f"  First row keys: {sorted(rows[0].keys())}")
            print(f"  First row: {rows[0]}")


if __name__ == "__main__":
    main()
