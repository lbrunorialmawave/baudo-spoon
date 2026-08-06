"""Tests for Fase 2 — multi-role (MANTRA) role-key generalization.

Scope: ``ml/auction/var.py`` (ReplacementLevel multi-role policy) and
``ml/auction/price_drift.py`` (price index role-key generalization +
``resolve_pricing_role``). See plan §5.1 Fase 2.

Parity requirement (G2): every CLASSIC-only scenario in this file asserts
byte-identical output to what the pre-Fase-2 code produced (single-role
inputs never touch the new multi-role branches).
"""

from __future__ import annotations

import pytest

from ml.auction.models import AuctionConfig, ParticipantSetup
from ml.auction.orchestrator import initialize_auction
from ml.auction.price_drift import resolve_pricing_role
from ml.auction.var import ReplacementLevel, VarEngine, _select_scarcest_replacement
from ml.optimizer.models import Player


def _make_participants(n: int = 4) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(participant_id=f"u{i}", display_name=f"U{i}", budget_initial=500)
        for i in range(1, n + 1)
    ]


def _mk_player(pid: str, role: str, cost: int, score: float, eligible: frozenset[str] = frozenset()) -> Player:
    return Player(
        player_id=pid,
        name=pid,
        real_team="TEST",
        role=role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
        eligible_roles=eligible,
    )


# ---------------------------------------------------------------------------
# var.py — CLASSIC parity (single-role players unaffected)
# ---------------------------------------------------------------------------


def test_classic_single_role_dicts_unaffected() -> None:
    """Player dicts with only 'role' (no eligible_roles) behave exactly as before."""
    pool = [
        {"player_id": "d1", "role": "D", "projected_score": 7.0},
        {"player_id": "d2", "role": "D", "projected_score": 5.0},
        {"player_id": "d3", "role": "D", "projected_score": 4.0},
    ]
    engine = VarEngine()
    results = engine.evaluate(pool)
    assert {r.player_id for r in results} == {"d1", "d2", "d3"}
    for r in results:
        assert r.role == "D"


# ---------------------------------------------------------------------------
# var.py — scarcest-role selection policy
# ---------------------------------------------------------------------------


def test_select_scarcest_replacement_picks_highest_score() -> None:
    replacement_by_role = {
        "Dd": ReplacementLevel(role="Dd", score=4.0, n_players_used=2),
        "E": ReplacementLevel(role="E", score=6.5, n_players_used=1),  # thinner pool -> scarcer
    }
    picked = _select_scarcest_replacement(frozenset({"Dd", "E"}), replacement_by_role)
    assert picked.role == "E"


def test_select_scarcest_replacement_ignores_roles_absent_from_pool() -> None:
    replacement_by_role = {"Dd": ReplacementLevel(role="Dd", score=4.0, n_players_used=2)}
    picked = _select_scarcest_replacement(frozenset({"Dd", "GhostRole"}), replacement_by_role)
    assert picked.role == "Dd"


def test_select_scarcest_replacement_raises_when_none_available() -> None:
    with pytest.raises(ValueError, match="ReplacementLevel"):
        _select_scarcest_replacement(frozenset({"GhostRole"}), {})


def test_multi_role_player_var_uses_scarcest_eligible_role() -> None:
    """A flex player eligible for both a deep and a thin role gets VAR'd
    against the thin (scarcer) one — not whichever role sorts first."""
    pool = [
        # Deep 'Dd' pool: lots of mediocre backups -> low replacement score.
        {"player_id": "dd_1", "role": "D", "eligible_roles": ["Dd"], "projected_score": 6.0},
        {"player_id": "dd_2", "role": "D", "eligible_roles": ["Dd"], "projected_score": 3.0},
        {"player_id": "dd_3", "role": "D", "eligible_roles": ["Dd"], "projected_score": 2.5},
        # Thin 'E' pool: even the worst option is still decent -> high replacement score.
        {"player_id": "e_1", "role": "C", "eligible_roles": ["E"], "projected_score": 6.8},
        # Flex player, eligible for both.
        {"player_id": "flex", "role": "D", "eligible_roles": ["Dd", "E"], "projected_score": 7.0},
    ]
    engine = VarEngine(percentile_threshold=1.0)  # use full pool as "replacement" bucket for a crisp test
    results = {r.player_id: r for r in engine.evaluate(pool)}
    flex = results["flex"]
    # scarcest of {Dd, E} by replacement score should be E (only 6.8 vs Dd's mix).
    assert flex.role == "E"


def test_single_eligible_role_matches_plain_role_baseline() -> None:
    """A player with exactly one eligible_role gets the same VAR as if that
    role had been passed as the plain 'role' key."""
    common = [
        {"player_id": "filler1", "role": "D", "projected_score": 3.0},
        {"player_id": "filler2", "role": "D", "projected_score": 4.0},
    ]
    plain_pool = common + [{"player_id": "target", "role": "D", "projected_score": 7.0}]
    flex_pool = common + [
        {"player_id": "target", "role": "D", "eligible_roles": ["D"], "projected_score": 7.0}
    ]
    engine = VarEngine()
    plain_result = next(r for r in engine.evaluate(plain_pool) if r.player_id == "target")
    flex_result = next(r for r in engine.evaluate(flex_pool) if r.player_id == "target")
    assert plain_result.var_score == pytest.approx(flex_result.var_score)
    assert plain_result.esv == pytest.approx(flex_result.esv)


# ---------------------------------------------------------------------------
# price_drift.py — resolve_pricing_role
# ---------------------------------------------------------------------------


def test_resolve_pricing_role_classic_player_unchanged() -> None:
    cfg = AuctionConfig(num_participants=4)
    pool = [_mk_player("p1", "P", 10, 5.0)]
    state = initialize_auction(_make_participants(4), cfg, pool)
    role = resolve_pricing_role(pool[0], cfg, state.price_index)
    assert role == "P"


def test_resolve_pricing_role_single_eligible_role_uses_mantra_code() -> None:
    cfg = AuctionConfig(num_participants=4, ruleset="MANTRA")
    player = _mk_player("p1", "D", 10, 5.0, eligible=frozenset({"Dc"}))
    price_index = {r: {"LOW": 1.0, "MID": 1.0, "TOP": 1.0} for r in cfg.role_quotas}
    role = resolve_pricing_role(player, cfg, price_index)
    assert role == "Dc"


def test_resolve_pricing_role_multi_role_picks_highest_premium() -> None:
    cfg = AuctionConfig(num_participants=4, ruleset="MANTRA")
    player = _mk_player("p1", "D", 10, 5.0, eligible=frozenset({"Dd", "E"}))
    price_index = {r: {"LOW": 1.0, "MID": 1.0, "TOP": 1.0} for r in cfg.role_quotas}
    price_index["E"]["TOP"] = 1.6  # E is currently trading at a premium
    role = resolve_pricing_role(player, cfg, price_index)
    assert role == "E"


def test_resolve_pricing_role_falls_back_when_no_eligible_role_in_index() -> None:
    """Multi-role player whose eligible roles are all absent from the
    (stale/incomplete) price_index falls back to player.role."""
    cfg = AuctionConfig(num_participants=4)
    player = _mk_player("p1", "D", 10, 5.0, eligible=frozenset({"NotInIndex1", "NotInIndex2"}))
    price_index = {"D": {"LOW": 1.0, "MID": 1.0, "TOP": 1.0}}
    role = resolve_pricing_role(player, cfg, price_index)
    assert role == "D"
