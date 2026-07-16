"""Fetch a fresh topstats.json sample from FotMob CDN for parser validation."""

from __future__ import annotations

from pathlib import Path

import httpx

URL = "https://data.fotmob.com/stats/55/season/27044/topstats.json"
DEST = Path(r"C:\Users\L.Brunori\Documents\Progetti\personal\analysis\response\topstats_27044.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=20.0, follow_redirects=True) as client:
        r = client.get(URL)
        r.raise_for_status()
    DEST.write_bytes(r.content)
    payload = r.json()
    top_lists = payload.get("TopLists") or []
    print(f"Wrote {len(r.content)} bytes to {DEST}")
    print(f"TopLists count: {len(top_lists)}")
    for i, tl in enumerate(top_lists):
        name = tl.get("StatName")
        rows = len(tl.get("StatList") or [])
        print(f"  [{i:>2}] StatName={name!r:<30} Category={tl.get('Category')!r:<14} rows={rows}")


if __name__ == "__main__":
    main()
