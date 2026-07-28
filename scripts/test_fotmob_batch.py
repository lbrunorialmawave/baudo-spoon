"""Batch test: compare FotMob suggest API vs DB matching for many players."""

from __future__ import annotations

import json
import urllib.request
import urllib.parse

PLAYERS = [
    "Rafael Leão",
    "Khephren Thuram",
    "De Ketelaere",
    "M'Bala Nzola",
    "Randal Kolo Muani",
    "Pafundi",
    "Zaniolo",
    "Baldanzi",
    "Soulé",
    "Gnonto",
    "Lapo Nava",
    "Tijjani Reijnders",
    "Yacine Adli",
    "Ismaël Bennacer",
    "Fikayo Tomori",
    "Malick Thiaw",
    "Karim Adeyemi",
    "Mathys Tel",
    "Warren Zaïre-Emery",
    "Eduardo Camavinga",
]

FOTMOB_SUGGEST_URL = (
    "https://www.fotmob.com/api/data/search/suggest"
    "?hits=5&lang=it,en,fr&term={}"
)


def fotmob_suggest(term: str) -> list[dict]:
    url = FOTMOB_SUGGEST_URL.format(urllib.parse.quote(term))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [
        s for g in data
        for s in g.get("suggestions", [])
        if s.get("type") == "player"
    ]


def main():
    print(f"{'Player':30s} {'Results':>4s}  {'Top result':45s}  {'Team'}")
    print("-" * 100)
    for p in PLAYERS:
        try:
            results = fotmob_suggest(p)
            count = len(results)
            if count > 0:
                top = results[0]
                print(
                    f"{p:30s} {count:>4d}  "
                    f"ID={top['id']:>8}  {top['name']:30s}  [{top.get('teamName','')}]"
                )
            else:
                print(f"{p:30s} {count:>4d}  {'NO RESULTS':45s}")
        except Exception as e:
            print(f"{p:30s} {'ERR':>4s}  {str(e):45s}")


if __name__ == "__main__":
    main()
