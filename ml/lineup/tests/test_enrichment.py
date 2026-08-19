"""Tests for matchday/hybrid EV enrichment."""

from __future__ import annotations

from ml.lineup.enrichment import (
    HybridInfo,
    MatchdayInfo,
    enrich_matched_players,
    parse_hybrid_rows,
    parse_matchday_rows,
)
from ml.roster_import.matcher import (
    CatalogPlayer,
    MatchStatus,
    MatchedPlayer,
)
from ml.roster_import.parser import ParsedPlayer


def _mp(
    name: str,
    fid: int,
    roles: tuple[str, ...] = ("C",),
    status: MatchStatus = MatchStatus.AUTO,
) -> MatchedPlayer:
    parsed = ParsedPlayer(name_raw=name, name_clean=name, cost=10, row_index=1)
    cat = CatalogPlayer(
        fantacalcio_id=fid,
        name=name,
        team="Inter",
        role_classic="C",
        roles_mantra=roles,
    )
    return MatchedPlayer(
        parsed=parsed,
        status=status,
        score=1.0,
        catalog=cat if status != MatchStatus.UNMATCHED else None,
    )


def test_parse_hybrid_and_matchday():
    hybrid = parse_hybrid_rows(
        [
            {"fantacalcio_id": 1, "fp_ibrido_voto": 7.2},
            {"fantacalcioId": 2, "FP_Ibrido": 65},  # 0-100 scale
            {"id": 3, "predicted_fantavoto": 6.5},
        ]
    )
    assert 1 in hybrid and abs(hybrid[1].fp_ibrido_voto - 7.2) < 1e-6
    assert 2 in hybrid
    assert 3 in hybrid

    md = parse_matchday_rows(
        [
            {
                "fantacalcio_id": 1,
                "probability": 0.9,
                "status": "starter",
                "opponent": "Cagliari",
            },
            {"fantacalcio_id": 2, "probability": 85, "status": "bench"},
        ]
    )
    assert md[1].probability == 0.9
    assert md[1].opponent_team == "Cagliari"
    assert md[2].probability == 85  # scaled later in enrichment


def test_enrich_uses_hybrid_and_matchday():
    players = [_mp("Barella", 1, ("C", "M")), _mp("Leao", 2, ("W", "A"))]
    hybrid = {
        1: HybridInfo(1, 7.5),
        2: HybridInfo(2, 7.8),
    }
    matchday = {
        1: MatchdayInfo(1, 0.95, "starter"),
        2: MatchdayInfo(2, 0.80, "starter", opponent_team="Lecce"),
    }
    cands, stats = enrich_matched_players(
        players, hybrid_by_fid=hybrid, matchday_by_fid=matchday
    )
    assert len(cands) == 2
    assert stats.with_hybrid == 2
    assert stats.with_matchday == 2
    assert stats.excluded_out == 0
    # Barella EV ≈ 7.5 * 0.95 * ~1.0
    barella = next(c for c in cands if c.name == "Barella")
    assert abs(barella.expected_value - 7.5 * 0.95) < 0.15
    assert "FP_Ibrido" in barella.breakdown_note


def test_injured_excluded():
    players = [_mp("Injured", 9, ("A",))]
    matchday = {9: MatchdayInfo(9, 0.0, "injured")}
    cands, stats = enrich_matched_players(players, matchday_by_fid=matchday)
    assert cands == []
    assert stats.excluded_out == 1


def test_unmatched_skipped():
    players = [_mp("X", 1, status=MatchStatus.UNMATCHED)]
    cands, stats = enrich_matched_players(players)
    assert cands == []
    assert stats.total == 0


def test_baseline_when_no_data():
    players = [_mp("Unknown", 99, ("Dc",))]
    cands, stats = enrich_matched_players(players)
    assert len(cands) == 1
    assert stats.baseline_fallback == 1
    assert stats.with_hybrid == 0
    assert cands[0].expected_value > 0
    assert "baseline" in cands[0].breakdown_note.lower() or "FP" in cands[0].breakdown_note
