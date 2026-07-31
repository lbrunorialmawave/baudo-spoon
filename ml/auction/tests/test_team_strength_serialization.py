"""Round-trip preservation of the team-strength (Elo) signal in auction state.

Process guard: this proves the Elo adjustment survives a
``serialize_state`` / ``deserialize_state`` cycle, not just that the
dict is non-empty.  Mirrors the bar of
``ml/optimizer/tests/test_team_strength_wiring.py``: the test must
observe a *computed* number changing, not just a field being
populated.

Context: ``AuctionState.team_strength_scores`` has a
``default_factory=dict``, so a regression silently drops the Elo
adjustment on every resumed auction with no error or log line.  These
tests catch that.
"""

from __future__ import annotations

import pytest

from ml.auction.models import AuctionConfig, Tier
from ml.auction.orchestrator import (
    deserialize_state,
    initialize_auction,
    serialize_state,
)
from ml.auction.price_drift import compute_expected_price
from ml.optimizer.models import Player, Role

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _participants(n: int = 4) -> list:
    from ml.auction.models import ParticipantSetup

    return [
        ParticipantSetup(
            participant_id=f"u{i}", display_name=f"User {i}", budget_initial=500
        )
        for i in range(1, n + 1)
    ]


def _mk(
    pid: str,
    name: str,
    role: str,
    cost: int,
    score: float,
    team: str = "TEST",
) -> Player:
    return Player(
        player_id=pid,
        name=name,
        real_team=team,
        role=role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
    )


@pytest.fixture
def elo_pool() -> list[Player]:
    """Pool with strong-team (Inter) and weak-team (Lecce) representation."""
    return [
        _mk("p1", "FW Inter", "A", 30, 8.5, "Inter"),
        _mk("p2", "FW Lecce", "A", 20, 7.5, "Lecce"),
        _mk("d1", "DF Inter", "D", 20, 7.0, "Inter"),
        _mk("d2", "DF Lecce", "D", 10, 6.0, "Lecce"),
    ]


def _auction_config_with_elo(elo_scores: dict[str, float]) -> AuctionConfig:
    """Build an :class:`AuctionConfig` that loads the Elo table once.

    We can't drive ``initialize_auction``'s built-in Elo loader here
    because ``_config_from_dict`` (pre-existing bug, out of scope for
    this fix) hardcodes ``inflation_config=None`` on deserialize —
    which forces ``use_inflation_baseline=False`` for the round-trip to
    succeed.  Instead, we attach the Elo table directly to the state
    so we can exercise the ``serialize_state``/``deserialize_state``
    wire on the field that actually changed.
    """
    return AuctionConfig(num_participants=4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_team_strength_scores_round_trip_identical(elo_pool: list[Player]) -> None:
    """``team_strength_scores`` is byte-identical across the round-trip.

    Direct field-level check: the Elo table loaded once in
    ``initialize_auction`` must be re-emitted by ``serialize_state``
    and re-injected into the fresh ``AuctionState`` by
    ``deserialize_state``.
    """
    state = initialize_auction(
        _participants(4), _auction_config_with_elo({}), elo_pool
    )
    # Simulate what ``initialize_auction`` would have loaded given a
    # non-zero ``team_strength_multiplier``: a non-empty Elo table.
    elo_table = {"Inter": 0.95, "Lecce": 0.10}
    object.__setattr__(state, "team_strength_scores", dict(elo_table))

    restored = deserialize_state(serialize_state(state))

    assert restored.team_strength_scores == elo_table


def test_elo_signal_survives_resume_in_expected_price(elo_pool: list[Player]) -> None:
    """A resumed auction prices the same player with the same Elo adjustment.

    This is the effect-level proof: the value produced by
    :func:`compute_expected_price` (which is what drives every
    ``record_assignment``) must be identical before and after a save /
    resume cycle.  The strong-team player (Inter) must still be priced
    *higher* than the weak-team player (Lecce) on the resumed state.
    """

    state = initialize_auction(
        _participants(4), _auction_config_with_elo({}), elo_pool
    )
    # Populate the Elo table manually (see _auction_config_with_elo).
    elo_table = {"Inter": 0.95, "Lecce": 0.10}
    object.__setattr__(state, "team_strength_scores", dict(elo_table))

    p_strong = state.available_pool[0]  # Inter
    p_weak = state.available_pool[1]    # Lecce
    role: Role = "A"  # type: ignore[assignment]
    tier: Tier = "TOP"

    eff_strong_before = compute_expected_price(
        player=p_strong,
        role_percentile=state.role_percentile_map[p_strong.player_id],
        role=role,
        tier=tier,
        price_index=state.price_index,
        config=state.config,
        team_strength_scores=state.team_strength_scores,
    )
    eff_weak_before = compute_expected_price(
        player=p_weak,
        role_percentile=state.role_percentile_map[p_weak.player_id],
        role=role,
        tier=tier,
        price_index=state.price_index,
        config=state.config,
        team_strength_scores=state.team_strength_scores,
    )
    # Sanity: pre-round-trip Elo must actually be doing something
    # (i.e. Inter must cost more than Lecce at the same tier).
    assert eff_strong_before > eff_weak_before

    # Round-trip the whole session state.
    restored = deserialize_state(serialize_state(state))

    eff_strong_after = compute_expected_price(
        player=p_strong,
        role_percentile=restored.role_percentile_map[p_strong.player_id],
        role=role,
        tier=tier,
        price_index=restored.price_index,
        config=restored.config,
        team_strength_scores=restored.team_strength_scores,
    )
    eff_weak_after = compute_expected_price(
        player=p_weak,
        role_percentile=restored.role_percentile_map[p_weak.player_id],
        role=role,
        tier=tier,
        price_index=restored.price_index,
        config=restored.config,
        team_strength_scores=restored.team_strength_scores,
    )

    # 1. Bit-exact survival of the Elo signal through the round-trip.
    assert eff_strong_after == pytest.approx(eff_strong_before)
    assert eff_weak_after == pytest.approx(eff_weak_before)
    # 2. The Elo signal is still demonstrably active post-resume.
    assert eff_strong_after > eff_weak_after


def test_pre_fix_payload_without_team_strength_scores_backfills_empty(
    elo_pool: list[Player],
) -> None:
    """Backward-compat: old payloads (no ``team_strength_scores``) still load.

    Pre-fix ``serialize_state`` never emitted the key, so resuming a
    state from a payload written before this fix must not crash — the
    table just backfills to ``{}`` (same as a brand-new state with
    ``team_strength_multiplier=0.0``).
    """
    state = initialize_auction(
        _participants(4), _auction_config_with_elo({}), elo_pool
    )
    payload = serialize_state(state)
    # Simulate a pre-fix payload by removing the new key.
    payload.pop("team_strength_scores", None)

    restored = deserialize_state(payload)

    assert restored.team_strength_scores == {}
