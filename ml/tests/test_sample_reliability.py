"""Unit tests for the sample_reliability package.

Covers PR1 (cohort classification), PR2 (sample weighting) and PR3
(per-90 shrinkage).  All tests are pure, deterministic and side-effect
free — no database, no model, no I/O.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.sample_reliability import (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
    DEFAULT_PRIOR_STRENGTH,
    SampleReliability,
    apply_shrinkage,
    classify_cohort,
    compute_sample_weight,
    estimate_prior_rate,
    profile_dataset,
)
from ml.sample_reliability.weights import (
    STRATEGY_BUCKETED,
    STRATEGY_CONSTANT,
    STRATEGY_LINEAR,
    STRATEGY_SQRT,
)


# ── PR1 — cohort classification ──────────────────────────────────────────────


class TestClassifyCohort:
    def test_below_hard_cutoff_is_insufficient(self) -> None:
        assert classify_cohort(0) == COHORT_INSUFFICIENT
        assert classify_cohort(50) == COHORT_INSUFFICIENT
        assert classify_cohort(99) == COHORT_INSUFFICIENT

    def test_at_hard_cutoff_is_limited(self) -> None:
        assert classify_cohort(100) == COHORT_LIMITED

    def test_at_standard_cutoff_is_standard(self) -> None:
        assert classify_cohort(800) == COHORT_STANDARD
        assert classify_cohort(2500) == COHORT_STANDARD

    def test_just_below_standard_is_limited(self) -> None:
        assert classify_cohort(799) == COHORT_LIMITED

    def test_none_and_nan_are_insufficient(self) -> None:
        assert classify_cohort(None) == COHORT_INSUFFICIENT
        assert classify_cohort(float("nan")) == COHORT_INSUFFICIENT
        assert classify_cohort(-10) == COHORT_INSUFFICIENT

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            classify_cohort(500, min_minutes_hard=900, standard_minutes=800)
        with pytest.raises(ValueError):
            classify_cohort(500, min_minutes_hard=-1, standard_minutes=800)


class TestSampleReliabilityDTO:
    def test_valid_construction(self) -> None:
        rel = SampleReliability(
            minutes=500, appearances=10, starts=4,
            cohort=COHORT_LIMITED, weight=0.65,
        )
        assert rel.minutes == 500
        assert rel.cohort == COHORT_LIMITED
        assert rel.weight == 0.65

    def test_invalid_cohort_rejected(self) -> None:
        with pytest.raises(ValueError):
            SampleReliability(
                minutes=500, appearances=10, starts=None,
                cohort="UNKNOWN", weight=0.5,
            )

    def test_weight_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            SampleReliability(
                minutes=500, appearances=10, starts=None,
                cohort=COHORT_LIMITED, weight=1.5,
            )
        with pytest.raises(ValueError):
            SampleReliability(
                minutes=500, appearances=10, starts=None,
                cohort=COHORT_LIMITED, weight=-0.1,
            )


class TestProfileDataset:
    def test_counts_match_thresholds(self) -> None:
        df = pd.DataFrame({"mins_played": [50, 100, 500, 800, 2000, None]})
        stats = profile_dataset(df)
        # 50 (insufficient), None (insufficient), 100 (limited),
        # 500 (limited), 800 (standard), 2000 (standard)
        assert stats["n_insufficient"] == 2
        assert stats["n_limited"] == 2
        assert stats["n_standard"] == 2
        assert stats["n_total"] == 6
        assert math.isclose(stats["share_standard"], 1 / 3, rel_tol=1e-9)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"mins_played": []})
        stats = profile_dataset(df)
        assert stats["n_total"] == 0
        assert stats["share_standard"] == 0.0

    def test_missing_minutes_column_raises(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(KeyError):
            profile_dataset(df)


# ── PR2 — sample weighting ───────────────────────────────────────────────────


class TestComputeSampleWeight:
    @pytest.mark.parametrize(
        "minutes,expected",
        [
            (0, 0.0),
            (50, 0.0),
            (99, 0.0),
            (100, math.sqrt(100 / 800)),
            (200, math.sqrt(200 / 800)),
            (400, math.sqrt(400 / 800)),
            (600, math.sqrt(600 / 800)),
            (799, math.sqrt(799 / 800)),
            (800, 1.0),
            (2500, 1.0),
        ],
    )
    def test_sqrt_strategy(self, minutes: int, expected: float) -> None:
        assert compute_sample_weight(minutes, strategy=STRATEGY_SQRT) == pytest.approx(expected)

    def test_constant_strategy(self) -> None:
        assert compute_sample_weight(100, strategy=STRATEGY_CONSTANT) == 1.0
        assert compute_sample_weight(799, strategy=STRATEGY_CONSTANT) == 1.0
        assert compute_sample_weight(800, strategy=STRATEGY_CONSTANT) == 1.0
        # Below hard cutoff still gets 0.0 — exclusion is independent of strategy.
        assert compute_sample_weight(99, strategy=STRATEGY_CONSTANT) == 0.0

    def test_linear_strategy(self) -> None:
        assert compute_sample_weight(100, strategy=STRATEGY_LINEAR) == pytest.approx(0.125)
        assert compute_sample_weight(400, strategy=STRATEGY_LINEAR) == pytest.approx(0.5)
        assert compute_sample_weight(800, strategy=STRATEGY_LINEAR) == 1.0

    def test_bucketed_strategy(self) -> None:
        # 100-399 bucket -> 0.35
        assert compute_sample_weight(200, strategy=STRATEGY_BUCKETED) == 0.35
        # 400-799 bucket -> 0.65
        assert compute_sample_weight(500, strategy=STRATEGY_BUCKETED) == 0.65
        # 800+ -> 1.0
        assert compute_sample_weight(900, strategy=STRATEGY_BUCKETED) == 1.0

    def test_none_and_negative_minutes_zero(self) -> None:
        assert compute_sample_weight(None) == 0.0
        assert compute_sample_weight(float("nan")) == 0.0
        assert compute_sample_weight(-50) == 0.0

    def test_weight_in_unit_interval(self) -> None:
        for m in range(0, 3001, 37):
            w = compute_sample_weight(m)
            assert 0.0 <= w <= 1.0

    def test_weight_monotonic_in_minutes(self) -> None:
        prev = compute_sample_weight(0)
        for m in range(1, 2001):
            current = compute_sample_weight(m)
            assert current >= prev - 1e-12
            prev = current

    def test_invalid_strategy_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_sample_weight(500, strategy="exponential")

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_sample_weight(
                500, standard_minutes=0,
            )
        with pytest.raises(ValueError):
            compute_sample_weight(
                500, min_minutes_hard=900, standard_minutes=800,
            )


# ── PR3 — shrinkage ─────────────────────────────────────────────────────────


class TestApplyShrinkage:
    def test_scalar(self) -> None:
        # observed 2.7 goals/90 from 100 minutes, prior 0.3, strength 300
        # -> (2.7 * 100 + 0.3 * 300) / (100 + 300) = (270 + 90) / 400 = 0.9
        adjusted = apply_shrinkage(
            2.7, minutes=100, prior_rate=0.3, prior_strength=300,
        )
        assert adjusted == pytest.approx(0.9)

    def test_large_sample_converges_to_observed(self) -> None:
        # With 100_000 minutes, the prior (strength 300) contributes only
        # ~0.3% residual error.  Use an absolute tolerance rather than
        # relative to make the assertion robust to the choice of base rate.
        adjusted = apply_shrinkage(
            1.2, minutes=100_000, prior_rate=0.3, prior_strength=300,
        )
        assert adjusted == pytest.approx(1.2, abs=5e-3)

    def test_zero_minutes_returns_prior(self) -> None:
        adjusted = apply_shrinkage(0.0, minutes=0, prior_rate=0.5, prior_strength=300)
        assert adjusted == pytest.approx(0.5)

    def test_series_input(self) -> None:
        rates = pd.Series([2.7, 1.2, 0.0])
        minutes = pd.Series([100, 5_000, 0])
        adjusted = apply_shrinkage(rates, minutes=minutes, prior_rate=0.3, prior_strength=300)
        assert isinstance(adjusted, pd.Series)
        assert adjusted.iloc[0] == pytest.approx(0.9)
        # (1.2 * 5000 + 0.3 * 300) / (5000 + 300) ≈ 1.149
        assert adjusted.iloc[1] == pytest.approx(1.149, rel=1e-4)
        assert adjusted.iloc[2] == pytest.approx(0.3)

    def test_invalid_prior_rejected(self) -> None:
        with pytest.raises(ValueError):
            apply_shrinkage(1.0, minutes=100, prior_rate=-0.1)
        with pytest.raises(ValueError):
            apply_shrinkage(1.0, minutes=100, prior_rate=0.1, prior_strength=-1)

    def test_negative_minutes_become_zero(self) -> None:
        # Defensive: negative denominator is treated as 0 → returns prior.
        adjusted = apply_shrinkage(1.0, minutes=-50, prior_rate=0.4, prior_strength=200)
        assert adjusted == pytest.approx(0.4)


class TestEstimatePriorRate:
    def test_uses_only_standard_cohort(self) -> None:
        rates = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        minutes = pd.Series([50, 200, 500, 800, 2500])
        prior = estimate_prior_rate(rates, minutes=minutes, min_minutes=800)
        # Standard cohort: rates [2.0, 1.0] -> median = 1.5
        assert prior == pytest.approx(1.5)

    def test_empty_cohort_returns_zero(self) -> None:
        rates = pd.Series([1.0, 2.0])
        minutes = pd.Series([100, 200])
        assert estimate_prior_rate(rates, minutes=minutes, min_minutes=800) == 0.0


# ── Smoke test: invariants required by plan.md §52 ──────────────────────────


def test_operational_invariant_limited_weight_lte_standard_weight() -> None:
    """For every minutes < 800, the sample weight must be <= the standard weight."""
    standard_w = compute_sample_weight(800)
    for m in range(0, 800):
        w = compute_sample_weight(m)
        assert w <= standard_w + 1e-12


def test_operational_invariant_cohorts_partition() -> None:
    """The three cohorts must partition the dataset (no overlap, no gaps)."""
    df = pd.DataFrame({"mins_played": list(range(0, 1500)) + [None]})
    stats = profile_dataset(df)
    assert stats["n_total"] == stats["n_insufficient"] + stats["n_limited"] + stats["n_standard"]


def test_default_prior_strength_is_positive() -> None:
    assert DEFAULT_PRIOR_STRENGTH > 0
