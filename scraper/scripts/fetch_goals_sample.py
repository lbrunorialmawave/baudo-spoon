"""Verify shape of an individual stat file (e.g. goals.json) from FotMob CDN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

URL = "https://data.fotmob.com/stats/55/season/27044/goals.json"
OUT = Path(__file__).resolve().parent.parent / "response" / "goals_27044.json"


def main() -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
    }
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as c:
        r = c.get(URL)
        r.raise_for_status()
        data = r.json()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    tl = data.get("TopLists", [])
    print(f"TopLists count: {len(tl)}")
    for i, top in enumerate(tl):
        if not isinstance(top, dict):
            continue
        sl = top.get("StatList") or []
        print(f"  [{i}] StatName={top.get('StatName')!r}  rows={len(sl)}")
        if sl and i == 0:
            first = sl[0]
            print(f"     first keys: {list(first.keys())}")
            print(f"     first record: {json.dumps(first, ensure_ascii=False)[:400]}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
