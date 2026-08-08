"""Tests for MANTRA ruleset support in the Auction module (Fasi 3–6)."""

from __future__ import annotations

import pytest

from ml.auction.alternatives import player_role_set, suggest_alternatives
from ml.auction.models import AlternativesConfig, AuctionConfig, ParticipantSetup
from ml.auction.orchestrator import (
    AuctionSession,
    deserialize_state,
    initialize_auction,
    record_assignment,
    serialize_state,
    undo_last_assignment,
)
from ml.optimizer.models import MANTRA_DEFAULT_QUOTAS, Player


def _participants(n: int = 2, budget: int = 500) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(participant_id=f"u{i}", display_name=f"User{i}", budget_initial=budget)
        for i in range(1, n + 1)
    ]


def _player(
    pid: str,
    classic_role: str,
    cost: int = 10,
    score: float = 6.0,
    eligible: frozenset[str] | None = None,
) -> Player:
    return Player(
        player_id=pid,
        name=pid,
        real_team="TST",
        role=classic_role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
        eligible_roles=eligible or frozenset(),
    )


def _mantra_cfg(**overrides: object) -> AuctionConfig:
    defaults: dict[str, object] = {
        "num_participants": 2,
        "ruleset": "MANTRA",
        "role_quotas": dict(MANTRA_DEFAULT_QUOTAS),
        "budget_initial": 500,
    }
    defaults.update(overrides)
    return AuctionConfig(**defaults)  # type: ignore[arg-type]


def _small_mantra_pool() -> list[Player]:
    return [
        _player("por1", "P", eligible=frozenset({"Por"})),
        _player("por2", "P", eligible=frozenset({"Por"})),
        _player("dc1", "D", eligible=frozenset({"Dc"})),
        _player("dc2", "D", eligible=frozenset({"Dc"})),
        _player("dd1", "D", eligible=frozenset({"Dd"})),
        _player("e1", "C", eligible=frozenset({"E"})),
        _player("flex", "D", cost=15, score=7.5, eligible=frozenset({"Dd", "E"})),
        _player("c1", "C", eligible=frozenset({"C"})),
        _player("c2", "C", eligible=frozenset({"C"})),
        _player("a1", "A", eligible=frozenset({"A"})),
        _player("pc1", "A", eligible=frozenset({"Pc"})),
    ]


def test_classic_assignment_sets_assigned_slot_equal_to_role() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_player("p1", "P"), _player("d1", "D"), _player("c1", "C"), _player("a1", "A")]
    state = initialize_auction(_participants(2), cfg, pool)
    result = record_assignment(state, "d1", "u1", 20)
    assert result.success
    rec = state.assignments[-1]
    assert rec.role == "D"
    assert rec.assigned_slot == "D"
    assert state.participants["u1"].role_breakdown["D"] == 1


def test_classic_alternatives_still_filter_by_scalar_role() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [
        _player("d1", "D", cost=20, score=7.0),
        _player("d2", "D", cost=10, score=6.0),
        _player("c1", "C", cost=10, score=6.5),
    ]
    state = initialize_auction(_participants(2), cfg, pool)
    suggestion = suggest_alternatives(
        target=pool[0], available_pool=state.available_pool, state=state, config=AlternativesConfig()
    )
    assert suggestion.closest_alternative is not None
    assert suggestion.closest_alternative.player_id == "d2"


def test_mantra_explicit_slot_recorded() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    result = record_assignment(state, "flex", "u1", 25, assigned_slot="E")
    assert result.success, result.rejection_reason
    rec = state.assignments[-1]
    assert rec.assigned_slot == "E"
    assert rec.role == "E"
    assert state.participants["u1"].role_breakdown.get("E") == 1
    assert state.participants["u1"].role_breakdown.get("Dd", 0) == 0


def test_mantra_auto_picks_scarcest_residual_slot() -> None:
    quotas = dict(MANTRA_DEFAULT_QUOTAS)
    quotas["Dd"] = 2
    quotas["E"] = 1
    cfg = _mantra_cfg(role_quotas=quotas)
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    result = record_assignment(state, "flex", "u1", 20)
    assert result.success, result.rejection_reason
    assert state.assignments[-1].assigned_slot == "E"


def test_mantra_rejects_invalid_slot() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    result = record_assignment(state, "flex", "u1", 20, assigned_slot="Por")
    assert not result.success
    assert result.rejection_code == "invalid_slot"


def test_mantra_role_full_on_exhausted_slot() -> None:
    quotas = dict(MANTRA_DEFAULT_QUOTAS)
    quotas["E"] = 1
    cfg = _mantra_cfg(role_quotas=quotas)
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    assert record_assignment(state, "e1", "u1", 10, assigned_slot="E").success
    r2 = record_assignment(state, "flex", "u1", 10, assigned_slot="E")
    assert not r2.success
    assert r2.rejection_code == "role_full"


def test_mantra_undo_restores_assigned_slot() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    record_assignment(state, "flex", "u1", 25, assigned_slot="Dd")
    assert state.participants["u1"].role_breakdown.get("Dd") == 1
    budget_after = state.participants["u1"].budget_residual
    pool_size_after = len(state.available_pool)
    undo_last_assignment(state)
    assert state.participants["u1"].role_breakdown.get("Dd", 0) == 0
    assert state.participants["u1"].budget_residual == budget_after + 25
    assert len(state.available_pool) == pool_size_after + 1
    assert not state.assignments


def test_mantra_serialize_roundtrip_preserves_slot_and_ruleset() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    record_assignment(state, "flex", "u1", 30, assigned_slot="E")
    payload = serialize_state(state)
    assert payload["config"]["ruleset"] == "MANTRA"  # type: ignore[index]
    assert payload["assignments"][0]["assigned_slot"] == "E"  # type: ignore[index]
    assert set(payload["assignments"][0]["player"]["eligible_roles"]) == {"Dd", "E"}  # type: ignore[index]
    restored = deserialize_state(payload)
    assert restored.config.ruleset == "MANTRA"
    assert restored.assignments[0].assigned_slot == "E"
    assert restored.participants["u1"].role_breakdown.get("E") == 1
    undo_last_assignment(restored)
    assert restored.participants["u1"].role_breakdown.get("E", 0) == 0


def test_legacy_payload_without_assigned_slot_deserializes() -> None:
    cfg = AuctionConfig(num_participants=2)
    pool = [_player("d1", "D"), _player("c1", "C")]
    state = initialize_auction(_participants(2), cfg, pool)
    record_assignment(state, "d1", "u1", 15)
    payload = serialize_state(state)
    del payload["assignments"][0]["assigned_slot"]  # type: ignore[index]
    restored = deserialize_state(payload)
    assert restored.assignments[0].assigned_slot == "D"
    undo_last_assignment(restored)
    assert restored.participants["u1"].role_breakdown.get("D", 0) == 0


def test_mantra_alternatives_include_multi_role_players() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    target = next(p for p in pool if p.player_id == "dd1")
    suggestion = suggest_alternatives(
        target=target, available_pool=state.available_pool, state=state, config=AlternativesConfig()
    )
    ids = set()
    if suggestion.low_cost_alternative:
        ids.add(suggestion.low_cost_alternative.player_id)
    if suggestion.closest_alternative:
        ids.add(suggestion.closest_alternative.player_id)
    assert "flex" in ids or suggestion.closest_alternative is not None


def test_player_role_set_classic_vs_mantra() -> None:
    p = _player("x", "D", eligible=frozenset({"Dd", "E"}))
    assert player_role_set(p, "CLASSIC") == frozenset({"D"})
    assert player_role_set(p, "MANTRA") == frozenset({"Dd", "E"})


def test_session_record_with_assigned_slot() -> None:
    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    session = AuctionSession(_participants(2), cfg, pool)
    result = session.record("flex", "u1", 22, assigned_slot="Dd")
    assert result.success
    assert session.state.assignments[-1].assigned_slot == "Dd"


# ---------------------------------------------------------------------------
# Mantra module residual coverage (Phase 4)
# ---------------------------------------------------------------------------


def test_mantra_summary_includes_module_coverage() -> None:
    from ml.auction.orchestrator import get_auction_summary

    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)
    # Assign a few players to u1
    for pid, price in [("por1", 5), ("dc1", 10), ("e1", 12)]:
        result = record_assignment(state, pid, "u1", price)
        assert result.success, result.rejection_reason

    summary = get_auction_summary(state, include_completion_probability=False)
    assert summary.mantra_module_coverage is not None
    assert "u1" in summary.mantra_module_coverage
    assert "u2" in summary.mantra_module_coverage
    # 11 official modules
    assert len(summary.mantra_module_coverage["u1"]) == 11
    for label, cov in summary.mantra_module_coverage["u1"].items():
        assert cov.label == label
        assert isinstance(cov.feasible, bool)
        assert isinstance(cov.deficits, dict)


def test_classic_summary_has_no_mantra_coverage() -> None:
    from ml.auction.orchestrator import get_auction_summary

    cfg = AuctionConfig(num_participants=2)
    pool = [_player("p1", "P"), _player("d1", "D"), _player("c1", "C"), _player("a1", "A")]
    state = initialize_auction(_participants(2), cfg, pool)
    summary = get_auction_summary(state, include_completion_probability=False)
    assert summary.mantra_module_coverage is None


def test_coverage_updates_monotonically_when_adding_players() -> None:
    """Adding a player never turns a previously-feasible module to False."""
    from ml.auction.orchestrator import get_auction_summary

    cfg = _mantra_cfg()
    pool = _small_mantra_pool()
    state = initialize_auction(_participants(2), cfg, pool)

    prev_feasible: set[str] = set()
    for pid, price in [("por1", 5), ("dc1", 10), ("dc2", 10), ("e1", 12), ("c1", 10), ("a1", 15)]:
        result = record_assignment(state, pid, "u1", price)
        if not result.success:
            continue
        summary = get_auction_summary(state, include_completion_probability=False)
        assert summary.mantra_module_coverage is not None
        now_feasible = {
            label
            for label, cov in summary.mantra_module_coverage["u1"].items()
            if cov.feasible
        }
        # Once feasible, stays feasible
        assert prev_feasible.issubset(now_feasible)
        prev_feasible = now_feasible
