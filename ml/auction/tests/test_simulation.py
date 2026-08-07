"""Tests for the Monte Carlo auction simulation engine."""
from __future__ import annotations
import pytest
from ml.auction.models import AuctionConfig, ParticipantSetup
from ml.auction.simulation import (
    AuctionSimulationConfig, BidderPolicy, BidderProfile, simulate_auction,
)
from ml.optimizer.models import Player

def _mk_player(pid, name, role, cost, score, team="TEST"):
    return Player(player_id=pid, name=name, real_team=team, role=role, cost=cost, projected_score=score)

def _small_pool():
    players = []
    for i in range(6): players.append(_mk_player(f"p{i}", f"GK {i}", "P", 5+i, 5.0+i*0.2))
    for i in range(10): players.append(_mk_player(f"d{i}", f"DF {i}", "D", 4+i, 5.0+i*0.15))
    for i in range(10): players.append(_mk_player(f"c{i}", f"MF {i}", "C", 6+i, 5.5+i*0.15))
    for i in range(8): players.append(_mk_player(f"a{i}", f"FW {i}", "A", 8+i, 6.0+i*0.2))
    return players

def _two_participants(budget=300):
    return [
        ParticipantSetup(participant_id="u0", display_name="User 0", budget_initial=budget),
        ParticipantSetup(participant_id="u1", display_name="User 1", budget_initial=budget),
    ]

def _config(n=2):
    return AuctionConfig(num_participants=n, role_quotas={"P":2,"D":3,"C":3,"A":2}, budget_initial=300, reference_budget=300)

def test_n_simulations_out_of_range():
    with pytest.raises(ValueError, match="n_simulations"): AuctionSimulationConfig(n_simulations=0)
    with pytest.raises(ValueError, match="n_simulations"): AuctionSimulationConfig(n_simulations=1001)

def test_determinism_same_seed():
    participants, pool, config = _two_participants(), _small_pool(), _config()
    profiles = [
        BidderProfile(participant_id="u0", policy=BidderPolicy(aggressiveness=0.4)),
        BidderProfile(participant_id="u1", policy=BidderPolicy(aggressiveness=0.6)),
    ]
    sim_cfg = AuctionSimulationConfig(n_simulations=5, random_seed=123)
    r1 = simulate_auction(participants, profiles, config, pool, sim_cfg)
    r2 = simulate_auction(participants, profiles, config, pool, sim_cfg)
    assert r1.n_completed == r2.n_completed == 5
    for pid in ("u0", "u1"):
        assert r1.per_participant[pid].spend_p50 == r2.per_participant[pid].spend_p50

def test_budget_residual_never_negative():
    result = simulate_auction(
        _two_participants(), [
            BidderProfile(participant_id="u0", policy=BidderPolicy(aggressiveness=0.9, max_overpay_ratio=1.5)),
            BidderProfile(participant_id="u1", policy=BidderPolicy(aggressiveness=0.9, max_overpay_ratio=1.5)),
        ], _config(), _small_pool(), AuctionSimulationConfig(n_simulations=10, random_seed=7),
    )
    assert not any("budget_residual < 0" in w for w in result.warnings)
    for s in result.per_participant.values():
        assert s.spend_p90 <= 300 + 1e-6

def test_single_bidder_completes():
    result = simulate_auction(
        [ParticipantSetup(participant_id="solo", display_name="Solo", budget_initial=500)],
        [BidderProfile(participant_id="solo", policy=BidderPolicy())],
        AuctionConfig(num_participants=1, role_quotas={"P":1,"D":2,"C":2,"A":1}, budget_initial=500, reference_budget=300),
        _small_pool(), AuctionSimulationConfig(n_simulations=8, random_seed=1),
    )
    assert result.n_completed == 8
    assert result.per_participant["solo"].completion_probability >= 0.5

def test_bidder_policy_validation():
    with pytest.raises(ValueError): BidderPolicy(aggressiveness=1.5)
    with pytest.raises(ValueError): BidderPolicy(max_overpay_ratio=0.5)
