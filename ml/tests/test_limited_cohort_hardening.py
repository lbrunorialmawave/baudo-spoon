"""Regression tests for LIMITED-cohort hardening (plan-limited-cohort-hardening.md).

Covers:
* WS0 canary dataset invariants
* WS2 continuous reliability weight properties (also covered in
  test_sample_reliability.py)
* Skeleton for WS1: after input-side shrinkage the Adzic-style raw
  per-90 must be pulled toward the prior (asserted via apply_shrinkage
  directly — full trainer path needs enable_shrinkage=True and is
  exercised in test_trainer_low_sample_flags).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.sample_reliability import (
    COHORT_LIMITED,
    COHORT_STANDARD,
    apply_shrinkage,
    continuous_reliability_weight,
    get_reliability_weight,
)
from ml.tests.fixtures.limited_cohort_canary import (
    CANARY_ANOMALY_IDS,
    build_limited_cohort_canary,
)


class TestCanaryInvariants:
    def test_canary_has_adzic_and_standard_refs(self) -> None:
        df = build_limited_cohort_canary()
        ids = set(df["player_id"])
        assert "adzic-163" in ids
        assert any(df["sample_cohort"] == COHORT_STANDARD)
        assert CANARY_ANOMALY_IDS.issubset(ids)

    def test_continuous_weight_orders_canary_correctly(self) -> None:
        df = build_limited_cohort_canary()
        df = df.copy()
        df["rel_w"] = df["mins_played"].map(continuous_reliability_weight)
        adzic_w = float(df.loc[df["player_id"] == "adzic-163", "rel_w"].iloc[0])
        near_w = float(df.loc[df["player_id"] == "lim-795", "rel_w"].iloc[0])
        std_w = float(df.loc[df["player_id"] == "std-fwd-1", "rel_w"].iloc[0])
        assert std_w == pytest.approx(1.0)
        assert near_w > adzic_w
        assert adzic_w < 0.55


class TestShrinkagePullsPhenomTowardPrior:
    """WS1 skeleton: pure shrinkage must damp extreme low-sample rates."""

    def test_adzic_style_per90_is_shrunk(self) -> None:
        # 1 goal in 163 minutes → raw per-90 ≈ 0.55
        raw_per90 = 1.0 / 163.0 * 90.0
        assert raw_per90 == pytest.approx(0.552, abs=0.01)

        # Prior ≈ realistic top Italian forward ~0.30
        prior = 0.30
        prior_strength = 300
        shrunk = apply_shrinkage(
            observed_rate=raw_per90,
            minutes=163,
            prior_rate=prior,
            prior_strength=prior_strength,
        )
        # Weight on observed = 163/(163+300) ≈ 0.352 → shrunk ≈ 0.39
        expected = (163 / (163 + 300)) * raw_per90 + (300 / (163 + 300)) * prior
        assert shrunk == pytest.approx(expected, abs=1e-9)
        assert shrunk < raw_per90
        assert shrunk < 0.45  # clearly below the raw phenom rate


class TestBucketVsContinuousDecisionWeight:
    def test_bucket_identical_for_105_and_795(self) -> None:
        assert get_reliability_weight(105, cohort=COHORT_LIMITED, mode="bucket") == (
            get_reliability_weight(795, cohort=COHORT_LIMITED, mode="bucket")
        )

    def test_continuous_differs_for_105_and_795(self) -> None:
        w105 = get_reliability_weight(105, mode="continuous")
        w795 = get_reliability_weight(795, mode="continuous")
        assert w795 > w105 + 0.05
