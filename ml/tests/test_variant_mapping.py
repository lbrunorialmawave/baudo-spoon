"""Tests for the shared FeatureFlag → harness-variant mapping (WS14-bis)."""

from __future__ import annotations

import pytest

from ml.rollout.controller import FeatureFlag
from ml.rollout.variant_mapping import (
    DEFAULT_VARIANT,
    FLAG_TO_VARIANT,
    variant_for_flag,
)


def test_variant_for_flag_shrinkage() -> None:
    assert variant_for_flag(FeatureFlag.PER90_SHRINKAGE.value) == "C_shrinkage"
    assert variant_for_flag("enable_shrinkage") == "C_shrinkage"


def test_variant_for_flag_limited_sample() -> None:
    assert (
        variant_for_flag(FeatureFlag.LIMITED_SAMPLE_TRAINING.value) == "B_weighting"
    )


def test_variant_for_flag_recent_role() -> None:
    assert (
        variant_for_flag(FeatureFlag.RECENT_ROLE_FEATURES.value)
        == "D_recent_role_features"
    )


def test_variant_for_flag_unmapped_raises() -> None:
    with pytest.raises(KeyError, match="no corresponding harness variant"):
        variant_for_flag(FeatureFlag.BREAKOUT_MODEL.value)
    with pytest.raises(KeyError, match="no corresponding harness variant"):
        variant_for_flag(FeatureFlag.RELIABILITY_WEIGHT_CONTINUOUS.value)
    with pytest.raises(KeyError):
        variant_for_flag("unknown_flag")


def test_flag_to_variant_keys_match_feature_flag_values() -> None:
    """Every key in the mapping must be a valid FeatureFlag value."""
    known = {f.value for f in FeatureFlag}
    for key in FLAG_TO_VARIANT:
        assert key in known


def test_default_variant_is_a_control() -> None:
    assert DEFAULT_VARIANT == "A_control"
