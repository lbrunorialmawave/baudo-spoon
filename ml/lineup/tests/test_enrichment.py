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


# ── Form blend / pre-match ───────────────────────────────────────────────────

from ml.lineup.enrichment import (
    blend_fp_with_form,
    filter_votes_pre_match,
    form_blend_weight,
)
from ml.trades.signals import MatchdayVote


def test_form_blend_weight_schedule():
    assert form_blend_weight(0) == 0.0
    assert form_blend_weight(1) == 0.15
    assert form_blend_weight(2) == 0.15
    assert form_blend_weight(3) == 0.25
    assert form_blend_weight(4) == 0.25
    assert form_blend_weight(5) == 0.35
    assert form_blend_weight(10) == 0.35


def test_filter_votes_pre_match_excludes_target_and_future():
    votes = [
        MatchdayVote(giornata=5, fantavoto=6.5),
        MatchdayVote(giornata=6, fantavoto=7.0),  # target
        MatchdayVote(giornata=7, fantavoto=8.0),  # future leak
        MatchdayVote(giornata=4, fantavoto=5.5),
    ]
    filtered = filter_votes_pre_match(votes, target_matchday=6)
    assert sorted(v.giornata for v in filtered) == [4, 5]
    # No target → no filter
    assert len(filter_votes_pre_match(votes, target_matchday=None)) == 4


def test_blend_fp_with_form_conservative():
    fp, lam, note = blend_fp_with_form(7.0, form_ewma=None, games_available=0)
    assert fp == 7.0 and lam == 0.0

    fp, lam, note = blend_fp_with_form(7.0, form_ewma=8.0, games_available=5)
    assert abs(lam - 0.35) < 1e-9
    # (1-0.35)*7 + 0.35*8 = 4.55 + 2.8 = 7.35
    assert abs(fp - 7.35) < 1e-9
    assert "form blend" in note


def test_enrich_blends_form_pre_match_only():
    """Target matchday 6: vote on giornata 6 must not move EV."""
    players = [_mp("Hot", 1, ("A",))]
    hybrid = {1: HybridInfo(1, 7.0)}
    matchday = {1: MatchdayInfo(1, 1.0, "starter")}
    # Strong vote only on target giornata — must be ignored pre-match
    votes = {
        1: [
            MatchdayVote(giornata=6, fantavoto=10.0),
            MatchdayVote(giornata=5, fantavoto=6.0),
        ]
    }
    cands, stats = enrich_matched_players(
        players,
        hybrid_by_fid=hybrid,
        matchday_by_fid=matchday,
        votes_by_fid=votes,
        target_matchday=6,
    )
    assert len(cands) == 1
    assert stats.with_form == 1
    # λ=0.15 (1 game), form=6.0 → fp_eff = 0.85*7 + 0.15*6 = 6.85
    # EV = 6.85 * 1.0 * adj(~1)
    assert abs(cands[0].expected_value - 6.85) < 0.2
    assert "form blend" in cands[0].breakdown_note

    # If only target-day vote exists → no form blend
    votes_only_target = {1: [MatchdayVote(giornata=6, fantavoto=10.0)]}
    cands2, stats2 = enrich_matched_players(
        players,
        hybrid_by_fid=hybrid,
        matchday_by_fid=matchday,
        votes_by_fid=votes_only_target,
        target_matchday=6,
    )
    assert stats2.with_form == 0
    assert abs(cands2[0].expected_value - 7.0) < 0.15
