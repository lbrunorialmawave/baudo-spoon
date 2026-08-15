#!/usr/bin/env python3
"""CI guard: ensure no duplicate ADR numbers among non-HISTORICAL docs."""

from __future__ import annotations

import sys
from pathlib import Path

ADR_DIR = Path(__file__).resolve().parents[2] / "docs" / "adr"


def main() -> int:
    if not ADR_DIR.is_dir():
        print(f"ERROR: ADR directory not found: {ADR_DIR}", file=sys.stderr)
        return 2
    by_num: dict[str, list[str]] = {}
    for p in sorted(ADR_DIR.glob("*.md")):
        if "HISTORICAL" in p.name.upper():
            continue
        num = p.name.split("-", 1)[0]
        if num.isdigit():
            by_num.setdefault(num, []).append(p.name)
    dupes = {k: v for k, v in by_num.items() if len(v) > 1}
    if dupes:
        print("ERROR: Duplicate ADR numbers detected:")
        for k, names in dupes.items():
            for n in names:
                print(f"  - {n}")
        return 1
    live_0001 = [n for n in by_num.get("0001", [])]
    if len(live_0001) != 1:
        print(f"ERROR: Expected exactly one live ADR 0001, found {live_0001}")
        return 1
    print(f"ADR uniqueness OK ({sum(len(v) for v in by_num.values())} live ADRs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
