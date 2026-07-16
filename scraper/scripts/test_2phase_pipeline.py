"""Validate the 2-phase stats pipeline with mocked HTTP."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats_scraper import (  # noqa: E402
    _fetch_full_season_stats,
    _discover_stat_urls,
    _infer_stat_type,
)


# ── Topstats payload (3 TopLists, solo top 3 per ciascuna) ──────────────────
TOPSTATS = {
    "TopLists": [
        {
            "StatName": "goals",
            "StatLocation": "https://data.fotmob.com/stats/55/season/27044/goals.json",
            "StatList": [
                {"ParticipantName": "A", "ParticiantId": 1, "TeamId": 10, "StatValue": 10.0, "Rank": 1},
                {"ParticipantName": "B", "ParticiantId": 2, "TeamId": 10, "StatValue": 9.0, "Rank": 2},
                {"ParticipantName": "C", "ParticiantId": 3, "TeamId": 11, "StatValue": 8.0, "Rank": 3},
            ],
        },
        {
            "StatName": "goals_team",
            "StatLocation": "https://data.fotmob.com/stats/55/season/27044/goals_team.json",
            "StatList": [
                {"ParticipantName": "Inter", "ParticiantId": 0, "TeamId": 8636, "StatValue": 50.0, "Rank": 1},
                {"ParticipantName": "Milan", "ParticiantId": 0, "TeamId": 8533, "StatValue": 45.0, "Rank": 2},
            ],
        },
        {
            "StatName": "rating",
            "StatLocation": "https://data.fotmob.com/stats/55/season/27044/rating.json",
            "StatList": [
                {"ParticipantName": "X", "ParticiantId": 99, "TeamId": 10, "StatValue": 7.5, "Rank": 1},
            ],
        },
    ]
}


# ── goals.json: 297 record reali (mockiamo solo 5) ──────────────────────────
GOALS_JSON = {
    "TopLists": [
        {
            "StatName": "goals",
            "StatList": [
                {"ParticipantName": f"P{i}", "ParticiantId": 1000 + i, "TeamId": 10, "StatValue": float(10 - i), "Rank": i + 1}
                for i in range(5)
            ],
        }
    ]
}


# ── goals_team.json: 20 squadre ──────────────────────────────────────────────
GOALS_TEAM_JSON = {
    "TopLists": [
        {
            "StatName": "goals_team",
            "StatList": [
                {"ParticipantName": f"Team{i}", "ParticiantId": 0, "TeamId": 8000 + i, "StatValue": float(50 - i), "Rank": i + 1}
                for i in range(20)
            ],
        }
    ]
}


# ── rating.json: 50 record ───────────────────────────────────────────────────
RATING_JSON = {
    "TopLists": [
        {
            "StatName": "rating",
            "StatList": [
                {"ParticipantName": f"R{i}", "ParticiantId": 2000 + i, "TeamId": 10, "StatValue": 7.0 - i * 0.01, "Rank": i + 1}
                for i in range(50)
            ],
        }
    ]
}


# ── Mock client ─────────────────────────────────────────────────────────────
class _MockResp:
    def __init__(self, data: dict, status: int = 200) -> None:
        self._data = data
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._data


def make_mock_client() -> AsyncMock:
    routes = {
        "topstats.json": _MockResp(TOPSTATS),
        "goals.json": _MockResp(GOALS_JSON),
        "goals_team.json": _MockResp(GOALS_TEAM_JSON),
        "rating.json": _MockResp(RATING_JSON),
    }
    client = AsyncMock()

    async def fake_get(url: str) -> _MockResp:
        name = url.rsplit("/", 1)[-1]
        if name in routes:
            return routes[name]
        return _MockResp({}, status=404)

    client.get = fake_get
    return client


# ── Test 1: _discover_stat_urls ──────────────────────────────────────────────
async def test_discover() -> None:
    client = make_mock_client()
    url = "https://data.fotmob.com/stats/55/season/27044/topstats.json"
    jobs = await _discover_stat_urls(client, url)
    assert len(jobs) == 3, f"atteso 3, ottenuto {len(jobs)}"
    assert jobs[0] == ("players", "goals", "https://data.fotmob.com/stats/55/season/27044/goals.json")
    assert jobs[1] == ("teams", "goals_team", "https://data.fotmob.com/stats/55/season/27044/goals_team.json")
    assert jobs[2] == ("players", "rating", "https://data.fotmob.com/stats/55/season/27044/rating.json")
    print("[OK] test_discover: 3 job estratti (2 players + 1 team)")


# ── Test 2: _infer_stat_type ────────────────────────────────────────────────
def test_infer_stat_type() -> None:
    # team list: ParticiantId=0, TeamId!=0
    team_entries = [
        {"ParticiantId": 0, "TeamId": 8636},
        {"ParticiantId": 0, "TeamId": 8533},
    ]
    assert _infer_stat_type("anything", team_entries, "players") == "teams"

    # player list
    player_entries = [
        {"ParticiantId": 1, "TeamId": 10},
        {"ParticiantId": 2, "TeamId": 10},
    ]
    assert _infer_stat_type("goals", player_entries, "players") == "players"

    # fallback suffisso
    mixed_or_empty = []
    assert _infer_stat_type("goals_team", mixed_or_empty, "players") == "teams"
    assert _infer_stat_type("goals_team_match", mixed_or_empty, "players") == "teams"
    assert _infer_stat_type("goals", mixed_or_empty, "players") == "players"
    print("[OK] test_infer_stat_type: tutte le regole coperte")


# ── Test 3: end-to-end _fetch_full_season_stats (mock) ──────────────────────
async def test_e2e_mock() -> None:
    from src.stats_scraper import _fetch_full_season_stats

    # Patch httpx.AsyncClient globally for the duration of this test
    import src.stats_scraper as m

    real_client = m.httpx.AsyncClient

    class FakeCtx:
        async def __aenter__(self) -> AsyncMock:
            return make_mock_client()

        async def __aexit__(self, *args) -> None:
            pass

    # We need to support kwargs (headers, timeout, limits, follow_redirects) used in _fetch_full_season_stats
    def fake_async_client(**kwargs) -> FakeCtx:  # noqa: ARG001
        return FakeCtx()

    m.httpx.AsyncClient = fake_async_client  # type: ignore[assignment]
    try:
        url = "https://data.fotmob.com/stats/55/season/27044/topstats.json"
        results = await _fetch_full_season_stats(url)
    finally:
        m.httpx.AsyncClient = real_client  # type: ignore[assignment]

    # Verifica shape: lista di (stat_type, stat_name, rows)
    assert len(results) == 3, f"atteso 3 tuple, ottenuto {len(results)}"
    by_name = {cat: (stype, rows) for stype, cat, rows in results}

    assert by_name["goals"][0] == "players"
    assert len(by_name["goals"][1]) == 5, "goals.json mock ha 5 record"

    assert by_name["goals_team"][0] == "teams"
    assert len(by_name["goals_team"][1]) == 20, "goals_team.json mock ha 20 record"

    assert by_name["rating"][0] == "players"
    assert len(by_name["rating"][1]) == 50, "rating.json mock ha 50 record"

    total_rows = sum(len(rows) for _, _, rows in results)
    print(f"[OK] test_e2e_mock: 3 file, {total_rows} record totali (topstats dava solo 6)")


# ── Test 4: end-to-end REALE (Serie A 2025-2026) ────────────────────────────
async def test_e2e_real() -> None:
    """Fetch reale di Serie A 2025-2026: verifica conteggio giocatori/squadre."""
    url = "https://data.fotmob.com/stats/55/season/27044/topstats.json"
    results = await _fetch_full_season_stats(url)
    if not results:
        print("[SKIP] test_e2e_real: fetch reale fallita (rete? rate-limit?)")
        return
    by_name = {cat: (stype, rows) for stype, cat, rows in results}
    player_total = sum(len(rows) for stype, _, rows in results if stype == "players")
    team_total = sum(len(rows) for stype, _, rows in results if stype == "teams")
    print(f"[OK] test_e2e_real: {len(results)} file, {player_total} players, {team_total} teams")
    # Verifica: goals deve avere ~250+ record (non 3)
    if "goals" in by_name:
        n = len(by_name["goals"][1])
        assert n > 100, f"goals: {n} record (atteso >100, topstats ne dava 3)"
        print(f"     goals.json: {n} record (topstats ne dava 3)")


async def main() -> None:
    test_infer_stat_type()
    await test_discover()
    await test_e2e_mock()
    await test_e2e_real()


if __name__ == "__main__":
    asyncio.run(main())
