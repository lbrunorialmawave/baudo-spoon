"""Unit tests for the production rollout module (PR8)."""

from __future__ import annotations

import pytest

from ml.rollout import (
    DEFAULT_ROLLOUT_PCT,
    FeatureFlag,
    FlagStage,
    RolloutController,
    default_controllers,
    shadow_compare,
)


class TestFeatureFlagEnum:
    def test_all_four_flags_defined(self) -> None:
        values = {flag.value for flag in FeatureFlag}
        # Each flag is mapped to an MLConfig field name.
        assert values == {
            "enable_limited_sample_training",
            "enable_shrinkage",
            "enable_recent_role_features",
            "enable_breakout_model",
        }


class TestShadowCompare:
    def test_identical_sequences_have_zero_delta(self) -> None:
        scores = [0.5, 0.6, 0.7, 0.8]
        out = shadow_compare(scores, scores)
        assert out.n_rows == 4
        assert out.absolute_delta == pytest.approx(0.0)
        assert out.relative_delta == pytest.approx(0.0, abs=1e-9)

    def test_computes_means_and_deltas(self) -> None:
        out = shadow_compare([0.0, 1.0], [0.2, 1.4])
        # baseline mean = 0.5, challenger mean = 0.8
        assert out.baseline_score == pytest.approx(0.5)
        assert out.challenger_score == pytest.approx(0.8)
        assert out.absolute_delta == pytest.approx(0.3)
        # rel_delta = 0.3 / 0.5 = 0.6
        assert out.relative_delta == pytest.approx(0.6)

    def test_empty_sequences_are_safe(self) -> None:
        out = shadow_compare([], [])
        assert out.n_rows == 0
        assert out.baseline_score == 0.0
        assert out.challenger_score == 0.0
        assert out.absolute_delta == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            shadow_compare([0.1, 0.2], [0.1])

    def test_includes_utc_timestamp(self) -> None:
        out = shadow_compare([0.1], [0.2])
        # UTC ISO format ends with "+00:00"
        assert out.timestamp.endswith("+00:00") or out.timestamp.endswith("Z")


class TestRolloutController:
    def test_default_controllers_are_all_disabled(self) -> None:
        controllers = default_controllers(random_seed=0)
        assert set(controllers.keys()) == set(FeatureFlag)
        for c in controllers.values():
            assert c.stage == FlagStage.DISABLED
            assert c.is_active() is False
            assert c.use_challenger() is False

    def test_disabled_never_uses_challenger(self) -> None:
        c = RolloutController(flag=FeatureFlag.LIMITED_SAMPLE_TRAINING)
        for _ in range(100):
            assert c.use_challenger() is False

    def test_shadow_never_uses_challenger(self) -> None:
        c = RolloutController(
            flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
            stage=FlagStage.SHADOW,
            rollout_pct=100.0,
            random_seed=0,
        )
        for _ in range(100):
            assert c.use_challenger() is False

    def test_active_uses_challenger_with_expected_frequency(self) -> None:
        c = RolloutController(
            flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
            stage=FlagStage.ACTIVE,
            rollout_pct=50.0,
            random_seed=42,
        )
        # Bernoulli with p=0.5 over 2000 samples → mean ≈ 0.5 ± 0.05.
        true_count = sum(c.use_challenger() for _ in range(2000))
        assert 0.45 < true_count / 2000 < 0.55

    def test_promotion_is_atomic(self) -> None:
        c = RolloutController(flag=FeatureFlag.PER90_SHRINKAGE)
        c.promote(new_stage=FlagStage.SHADOW)
        assert c.stage == FlagStage.SHADOW
        c.promote(new_stage=FlagStage.ACTIVE, new_rollout_pct=25.0)
        assert c.stage == FlagStage.ACTIVE
        assert c.rollout_pct == 25.0
        # Events were appended.
        assert len(c.events) == 2
        assert c.events[0]["from_stage"] == "disabled"
        assert c.events[1]["to_stage"] == "active"

    def test_rollback_emits_emergency_reason(self) -> None:
        c = RolloutController(
            flag=FeatureFlag.BREAKOUT_MODEL,
            stage=FlagStage.ACTIVE,
            rollout_pct=50.0,
        )
        c.promote(new_stage=FlagStage.DISABLED)
        assert c.events[-1]["reason"] == "emergency_rollback"
        assert c.events[-1]["from_stage"] == "active"
        assert c.events[-1]["to_stage"] == "disabled"

    def test_invalid_rollout_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            RolloutController(
                flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
                rollout_pct=-1.0,
            )
        with pytest.raises(ValueError):
            RolloutController(
                flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
                rollout_pct=101.0,
            )

    def test_active_with_zero_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            RolloutController(
                flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
                stage=FlagStage.ACTIVE,
                rollout_pct=0.0,
            )

    def test_promote_with_invalid_pct_rejected(self) -> None:
        c = RolloutController(flag=FeatureFlag.LIMITED_SAMPLE_TRAINING)
        with pytest.raises(ValueError):
            c.promote(new_stage=FlagStage.SHADOW, new_rollout_pct=200.0)


class TestDefaultRolloutPct:
    def test_default_pct_is_positive(self) -> None:
        assert DEFAULT_ROLLOUT_PCT > 0
        assert DEFAULT_ROLLOUT_PCT < 100
