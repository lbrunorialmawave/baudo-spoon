"""Golden tests for credit penalty (plan §9.3)."""

from __future__ import annotations

import pytest

from ml.trades.credit_penalty import recompute_value_on_transfer, round_half_up


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.75, 2),
        (2.25, 2),
        (0.5, 1),
        (0.49, 0),
        (100.0, 100),
    ],
)
def test_round_half_up(value, expected):
    assert round_half_up(value) == expected


def test_single_transfer_standard():
    assert recompute_value_on_transfer(100, 100, 25.0, 25.0) == 75


def test_three_successive_transfers():
    v = 100
    for expected in (75, 50, 25):
        v = recompute_value_on_transfer(100, v, 25.0, 25.0)
        assert v == expected
    # further transfers stay at floor
    assert recompute_value_on_transfer(100, v, 25.0, 25.0) == 25


def test_already_at_floor_unchanged():
    assert recompute_value_on_transfer(100, 25, 25.0, 25.0) == 25


def test_rounding_point_five():
    # 25% of 7 = 1.75 → step 2; new = 5
    assert recompute_value_on_transfer(7, 7, 25.0, 25.0) == 5


def test_rounding_below_half():
    # 25% of 9 = 2.25 → step 2; new = 7
    assert recompute_value_on_transfer(9, 9, 25.0, 25.0) == 7


def test_very_low_original():
    # 25% of 2 = 0.5 → step 1; floor = max(1, 1) = 1
    assert recompute_value_on_transfer(2, 2, 25.0, 25.0) == 1


def test_differentiated_floor():
    v = 100
    # step 25, floor 10
    v = recompute_value_on_transfer(100, v, 25.0, 10.0)
    assert v == 75
    v = recompute_value_on_transfer(100, v, 25.0, 10.0)
    assert v == 50
    v = recompute_value_on_transfer(100, v, 25.0, 10.0)
    assert v == 25
    v = recompute_value_on_transfer(100, v, 25.0, 10.0)
    assert v == 10
    v = recompute_value_on_transfer(100, v, 25.0, 10.0)
    assert v == 10


def test_zero_original():
    assert recompute_value_on_transfer(0, 0, 25.0, 25.0) == 0
