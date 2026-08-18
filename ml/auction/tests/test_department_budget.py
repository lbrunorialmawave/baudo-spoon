"""Tests for pre-auction department spending ceilings."""
from __future__ import annotations
import pytest
from ml.auction.department_budget import (
    CLASSIC_DEPARTMENTS, LISTINO_BUDGET_SHARE_PRIOR, MANTRA_DEPARTMENTS,
    MANTRA_LISTINO_BUDGET_SHARE_PRIOR, DepartmentCapConfig, compute_department_budget_plan,
)
from ml.auction.models import AuctionConfig
from ml.optimizer.models import MANTRA_DEFAULT_QUOTAS

def _classic(budget=500):
    return AuctionConfig(num_participants=8, role_quotas={"P":3,"D":8,"C":8,"A":6},
                         ruleset="CLASSIC", budget_initial=budget, reference_budget=300)
def _mantra(budget=500):
    return AuctionConfig(num_participants=8, role_quotas=dict(MANTRA_DEFAULT_QUOTAS),
                         ruleset="MANTRA", budget_initial=budget, reference_budget=300)

def test_classic_hard_and_recommended():
    plan = compute_department_budget_plan(_classic(500))
    by = {d.department_id: d for d in plan.departments}
    assert by["P"].hard_cap.credits == 478
    assert by["A"].hard_cap.credits == 481
    assert by["P"].recommended_max.credits == 60
    assert by["A"].recommended_max.credits == 175
    for d in plan.departments:
        assert d.market_share_source == "listino_prior"
        assert d.recommended_max.credits <= d.hard_cap.credits

def test_mantra_uses_calibrated_prior():
    plan = compute_department_budget_plan(_mantra(500))
    by = {d.department_id: d for d in plan.departments}
    for d in plan.departments:
        assert d.market_share_source == "listino_prior"
        assert d.market_share_prior == MANTRA_LISTINO_BUDGET_SHARE_PRIOR[d.department_id]
    assert by["POR"].recommended_max.credits == 63
    assert by["DIF"].recommended_max.credits == 153
    assert by["ATT"].recommended_max.credits == 128

def test_slot_share_sums_to_one():
    for cfg in (_classic(), _mantra()):
        assert abs(sum(d.slot_share for d in compute_department_budget_plan(cfg).departments) - 1.0) < 1e-6

def test_recommended_never_exceeds_hard():
    for b in (50, 300, 500):
        for f in (_classic, _mantra):
            for d in compute_department_budget_plan(f(b)).departments:
                assert d.recommended_max.credits <= d.hard_cap.credits

def test_extreme_low_budget_clamps():
    plan = compute_department_budget_plan(_classic(25))
    assert any(d.clamped_to_hard_cap for d in plan.departments)

def test_priors_sum_to_one():
    assert abs(sum(LISTINO_BUDGET_SHARE_PRIOR.values()) - 1.0) < 1e-9
    assert abs(sum(MANTRA_LISTINO_BUDGET_SHARE_PRIOR.values()) - 1.0) < 1e-9

def test_min_slot_price():
    p1 = compute_department_budget_plan(_classic(500), DepartmentCapConfig(min_slot_price=1))
    p2 = compute_department_budget_plan(_classic(500), DepartmentCapConfig(min_slot_price=2))
    assert {d.department_id: d for d in p1.departments}["A"].hard_cap.credits == 481
    assert {d.department_id: d for d in p2.departments}["A"].hard_cap.credits == 462

def test_config_validation():
    with pytest.raises(ValueError): DepartmentCapConfig(min_slot_price=0)
    with pytest.raises(ValueError): AuctionConfig(num_participants=2, min_slot_price=0)
