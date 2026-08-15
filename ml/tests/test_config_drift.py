"""Tests for ml.rollout.config_drift (WS5 of plan.md)."""

from __future__ import annotations

import json

import pytest

from ml.rollout.config_drift import (
    DriftFinding,
    DriftReport,
    DriftSeverity,
    DriftType,
    EffectiveConfig,
    RolloutSnapshot,
    detect_config_drift,
    effective_config_from_mapping,
    merge_reports,
    render_markdown,
    rollout_snapshot_from_resolved,
)
from ml.rollout.controller import FeatureFlag, FlagStage
from ml.rollout.env_flags import ResolvedFlags


# ── Helpers ─────────────────────────────────────────────────────────────────


def _snapshot(
    stage: FlagStage = FlagStage.DISABLED,
    mode: str = "bucket",
    *,
    production: dict[str, bool] | None = None,
    challenger: dict[str, bool] | None = None,
) -> RolloutSnapshot:
    return RolloutSnapshot(
        stage=stage,
        production_mode=mode,
        production_flags=production or {},
        challenger_flags=challenger or {},
    )


def _effective(
    mode: str = "bucket",
    *,
    use_new: bool = False,
    production: dict[str, bool] | None = None,
    challenger_enabled: bool = False,
) -> EffectiveConfig:
    return EffectiveConfig(
        production_mode=mode,
        use_new_behavior=use_new,
        production_flags=production or {},
        challenger_enabled=challenger_enabled,
    )


# ── Constructor validation ─────────────────────────────────────────────────


class TestSnapshotValidation:
    def test_valid_modes_accepted(self) -> None:
        for mode in ("bucket", "continuous"):
            s = _snapshot(mode=mode)
            assert s.production_mode == mode

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="bucket|continuous"):
            _snapshot(mode="true")
        with pytest.raises(ValueError, match="bucket|continuous"):
            _snapshot(mode="BUCKET")  # not normalised
        with pytest.raises(ValueError, match="bucket|continuous"):
            _snapshot(mode="")

    def test_stage_must_be_flag_stage(self) -> None:
        with pytest.raises(ValueError, match="FlagStage"):
            RolloutSnapshot(stage="active", production_mode="bucket")  # type: ignore[arg-type]

    def test_effective_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="bucket|continuous"):
            _effective(mode="continuous_false")


# ── Drift detection: stage ──────────────────────────────────────────────────


class TestStageMismatch:
    def test_active_requires_use_new_true(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="continuous", use_new=False),
        )
        assert report.has_p0
        assert any(f.drift_type == DriftType.STAGE_MISMATCH for f in report.findings)
        assert report.exit_code() == 1

    def test_disabled_requires_use_new_false(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.DISABLED, mode="bucket"),
            _effective(mode="bucket", use_new=True),
        )
        assert report.has_p0
        assert any(f.drift_type == DriftType.STAGE_MISMATCH for f in report.findings)

    def test_shadow_keeps_production_off(self) -> None:
        # SHADOW: use_new_behavior must be False, but challenger should run.
        report = detect_config_drift(
            _snapshot(stage=FlagStage.SHADOW, mode="bucket"),
            _effective(
                mode="bucket", use_new=False, challenger_enabled=True
            ),
        )
        assert not report.has_p0
        assert not report.has_drift  # no mode mismatch expected

    def test_shadow_use_new_true_is_drift(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.SHADOW, mode="bucket"),
            _effective(mode="bucket", use_new=True),
        )
        assert report.has_p0
        assert any(f.drift_type == DriftType.STAGE_MISMATCH for f in report.findings)


# ── Drift detection: mode ──────────────────────────────────────────────────


class TestModeMismatch:
    def test_active_continuous_matches(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="continuous", use_new=True),
        )
        assert not report.has_drift

    def test_active_with_bucket_effective_is_p0(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=True),
        )
        assert report.has_p0
        assert any(f.drift_type == DriftType.MODE_MISMATCH for f in report.findings)

    def test_disabled_bucket_matches(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.DISABLED, mode="bucket"),
            _effective(mode="bucket", use_new=False),
        )
        assert not report.has_drift

    def test_shadow_production_bucket_allowed(self) -> None:
        # SHADOW must keep production on bucket even though challenger is continuous.
        report = detect_config_drift(
            _snapshot(stage=FlagStage.SHADOW, mode="bucket"),
            _effective(mode="bucket", use_new=False, challenger_enabled=True),
        )
        assert not report.has_p0

    def test_invalid_rollout_mode(self) -> None:
        # Constructed via object.__setattr__ bypass to simulate bad payload.
        s = _snapshot(mode="bucket")
        object.__setattr__(s, "production_mode", "foo")
        report = detect_config_drift(s, _effective(mode="bucket"))
        assert report.has_p0
        assert any(
            f.drift_type == DriftType.INVALID_VALUE
            and f.field == "rollout.production_mode"
            for f in report.findings
        )


# ── Drift detection: boolean flags ─────────────────────────────────────────


class TestBooleanMismatch:
    def test_matching_flag_no_drift(self) -> None:
        report = detect_config_drift(
            _snapshot(
                stage=FlagStage.ACTIVE,
                mode="continuous",
                production={"enable_shrinkage": True},
            ),
            _effective(
                mode="continuous",
                use_new=True,
                production={"enable_shrinkage": True},
            ),
        )
        assert not report.has_drift

    def test_flag_true_in_rollout_false_in_effective(self) -> None:
        report = detect_config_drift(
            _snapshot(
                stage=FlagStage.ACTIVE,
                mode="continuous",
                production={"enable_shrinkage": True},
            ),
            _effective(
                mode="continuous",
                use_new=True,
                production={"enable_shrinkage": False},
            ),
        )
        assert report.has_p0
        assert any(
            f.drift_type == DriftType.BOOLEAN_MISMATCH
            and f.field == "effective.production_flags['enable_shrinkage']"
            for f in report.findings
        )

    def test_missing_flag_emits_misssing_field(self) -> None:
        report = detect_config_drift(
            _snapshot(
                stage=FlagStage.ACTIVE,
                mode="continuous",
                production={"enable_shrinkage": True},
            ),
            _effective(mode="continuous", use_new=True, production={}),
        )
        # Missing required field is P1, not P0.
        assert not report.has_p0
        assert report.has_p1
        assert any(f.drift_type == DriftType.MISSING_FIELD for f in report.findings)


# ── Drift detection: challenger ────────────────────────────────────────────


class TestChallengerMismatch:
    def test_shadow_without_challenger_is_p1(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.SHADOW, mode="bucket"),
            _effective(mode="bucket", use_new=False, challenger_enabled=False),
        )
        assert not report.has_p0
        assert report.has_p1
        assert any(
            f.drift_type == DriftType.CHALLENGER_MISMATCH for f in report.findings
        )

    def test_disabled_with_challenger_is_p1(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.DISABLED, mode="bucket"),
            _effective(mode="bucket", use_new=False, challenger_enabled=True),
        )
        assert not report.has_p0
        assert report.has_p1

    def test_active_without_challenger_is_p2(self) -> None:
        # Informational; ACTIVE doesn't strictly require challenger_enabled flag.
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="continuous", use_new=True, challenger_enabled=False),
        )
        # No P0/P1 from challenger side.  This is a soft signal — the
        # promotion gate is the canonical blocker.
        assert not report.has_p0
        assert not report.has_p1


# ── Report utilities ────────────────────────────────────────────────────────


class TestReportAPI:
    def test_exit_code_zero_when_clean(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.DISABLED, mode="bucket"),
            _effective(mode="bucket", use_new=False),
        )
        assert report.exit_code() == 0
        assert not report.has_drift
        assert report.highest_severity is None

    def test_highest_severity_p0(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=True),
        )
        assert report.highest_severity == DriftSeverity.P0

    def test_to_dict_serialisable(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=True),
        )
        d = report.to_dict()
        # Round-trip JSON
        s = json.dumps(d)
        loaded = json.loads(s)
        assert "findings" in loaded
        assert loaded["exit_code"] == 1

    def test_render_markdown_contains_severity_table(self) -> None:
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=False),
        )
        md = render_markdown(report)
        assert "# Configuration drift report" in md
        assert "| Severity | Type | Field |" in md
        assert "stage_mismatch" in md

    def test_merge_reports_combines_findings(self) -> None:
        a = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="continuous", use_new=True),
        )
        b = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=True),
        )
        merged = merge_reports([a, b])
        assert merged.has_p0
        assert any(f.drift_type == DriftType.MODE_MISMATCH for f in merged.findings)

    def test_merge_reports_empty(self) -> None:
        merged = merge_reports([])
        assert not merged.has_drift
        assert merged.exit_code() == 0


# ── Snapshot builders ─────────────────────────────────────────────────────


class TestBuilders:
    def test_rollout_snapshot_from_resolved_disabled(self) -> None:
        resolved = ResolvedFlags(
            production={"enable_shrinkage": False},
            challenger={"enable_shrinkage": False},
            stages={"enable_shrinkage": FlagStage.DISABLED.value},
            reliability_weight_mode="bucket",
        )
        snap = rollout_snapshot_from_resolved(resolved)
        assert snap.stage == FlagStage.DISABLED
        assert snap.production_mode == "bucket"

    def test_rollout_snapshot_from_resolved_active(self) -> None:
        resolved = ResolvedFlags(
            production={"enable_shrinkage": True},
            challenger={"enable_shrinkage": True},
            stages={"enable_shrinkage": FlagStage.ACTIVE.value},
            reliability_weight_mode="continuous",
        )
        snap = rollout_snapshot_from_resolved(resolved)
        assert snap.stage == FlagStage.ACTIVE
        assert snap.production_mode == "continuous"

    def test_effective_config_from_mapping(self) -> None:
        eff = effective_config_from_mapping(
            {
                "production_mode": "continuous",
                "use_new_behavior": True,
                "production_flags": {"enable_shrinkage": True},
                "challenger_enabled": False,
            }
        )
        assert eff.production_mode == "continuous"
        assert eff.use_new_behavior is True
        assert eff.production_flags == {"enable_shrinkage": True}

    def test_effective_config_from_mapping_falls_back(self) -> None:
        eff = effective_config_from_mapping(
            {
                "reliability_weight_mode": "bucket",
                "compute_new_behavior": True,
            }
        )
        assert eff.production_mode == "bucket"
        assert eff.use_new_behavior is True


# ── Mandatory negative tests from plan §23 ────────────────────────────────


class TestMandatoryNegative:
    def test_mode_equals_true_rejected(self) -> None:
        """mode=true must fail validation."""
        with pytest.raises(ValueError):
            _snapshot(mode="true")

    def test_mode_equals_false_rejected(self) -> None:
        with pytest.raises(ValueError):
            _snapshot(mode="false")

    def test_mode_equals_foo_rejected(self) -> None:
        with pytest.raises(ValueError):
            _snapshot(mode="foo")

    def test_runtime_rollout_mode_mismatch_denies_promotion(self) -> None:
        """Scenario E — runtime drift = activation DENY."""
        report = detect_config_drift(
            _snapshot(stage=FlagStage.ACTIVE, mode="continuous"),
            _effective(mode="bucket", use_new=True),
        )
        # Any P0 forces exit 1 → promotion gate sees DENY.
        assert report.exit_code() == 1


# ── End-to-end with real FeatureFlag enum ──────────────────────────────────


class TestRealFlagEnum:
    def test_shrinkage_flag_active_drift(self) -> None:
        report = detect_config_drift(
            RolloutSnapshot(
                stage=FlagStage.ACTIVE,
                production_mode="continuous",
                production_flags={FeatureFlag.PER90_SHRINKAGE.value: True},
            ),
            _effective(
                mode="continuous",
                use_new=True,
                production={FeatureFlag.PER90_SHRINKAGE.value: False},
            ),
        )
        assert report.has_p0
        assert any(
            f.drift_type == DriftType.BOOLEAN_MISMATCH for f in report.findings
        )
