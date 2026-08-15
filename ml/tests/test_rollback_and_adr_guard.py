"""Rollback integration tests (WS11) and ADR uniqueness CI guard (WS2)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ML_DATABASE_URL", "postgresql://x:x@localhost/x")

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rollback_active_to_disabled_restores_legacy_path():
    from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController

    ctrl = RolloutController(
        flag=FeatureFlag.PER90_SHRINKAGE,
        stage=FlagStage.ACTIVE,
        rollout_pct=100.0,
        random_seed=7,
    )
    assert ctrl.use_challenger() is True

    # Emergency rollback
    ctrl.promote(new_stage=FlagStage.DISABLED, new_rollout_pct=0.0)
    assert ctrl.stage == FlagStage.DISABLED
    assert ctrl.use_challenger() is False
    assert ctrl.is_active() is False


def test_rollback_continuous_to_bucket_mode():
    from ml.rollout.controller import FlagStage, reliability_weight_mode_for_stage
    from ml.sample_reliability import get_reliability_weight

    minutes = 163
    active_mode = reliability_weight_mode_for_stage(FlagStage.ACTIVE)
    disabled_mode = reliability_weight_mode_for_stage(FlagStage.DISABLED)
    assert active_mode == "continuous"
    assert disabled_mode == "bucket"

    w_active = get_reliability_weight(minutes=minutes, mode=active_mode)
    w_legacy = get_reliability_weight(minutes=minutes, mode=disabled_mode)
    # Both valid; rollback is the mode switch itself
    assert 0.0 < w_active <= 1.0
    assert 0.0 < w_legacy <= 1.0


def test_rollback_apply_reliability_weight_kill_switch():
    from ml.auction.decision_score import compute_decision_score
    from ml.sample_reliability import continuous_reliability_weight

    rw = continuous_reliability_weight(163)
    with_weight = compute_decision_score(
        projected_score=7.5, reliability_weight=rw, apply_reliability_weight=True
    )
    kill_switch = compute_decision_score(
        projected_score=7.5, reliability_weight=rw, apply_reliability_weight=False
    )
    assert with_weight < kill_switch
    assert kill_switch == pytest.approx(7.5)


def test_gate_failure_implies_no_active_promotion(tmp_path):
    """Simulated gate failure → stay DISABLED (rollback posture)."""
    import json
    from ml.scripts.check_promotion_gate import main
    from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController

    report = {
        "variants": {
            "A_control": {"status": "ok", "mae": 1.0},
            "C_shrinkage": {
                "status": "ok",
                "mae": 1.0,
                "rmse": 1.2,
                "mae_by_cohort": {"STANDARD": 0.9, "LIMITED": 1.0},
                "rmse_by_cohort": {"STANDARD": 1.1, "LIMITED": 1.2},
                "phenom_leakage_rate": 0.05,
                "canary_anomalies_remaining": 2,  # FAIL
            },
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(path)]) == 1

    # Operator must not promote
    ctrl = RolloutController(flag=FeatureFlag.PER90_SHRINKAGE, stage=FlagStage.SHADOW)
    # Stay at SHADOW / DISABLED — do not call promote to ACTIVE
    assert ctrl.use_challenger() is False


def test_adr_0001_uniqueness():
    """CI guard: exactly one canonical ADR 0001 file (not HISTORICAL)."""
    adr_dir = REPO_ROOT / "docs" / "adr"
    assert adr_dir.is_dir(), "docs/adr missing"
    canonical = list(adr_dir.glob("0001-*.md"))
    # Filter historical
    live = [p for p in canonical if "HISTORICAL" not in p.name.upper()]
    assert len(live) == 1, f"Expected exactly one live ADR 0001, found {[p.name for p in live]}"
    assert "auction-reliability" in live[0].name or "reliability" in live[0].name


def test_no_duplicate_adr_numbers():
    adr_dir = REPO_ROOT / "docs" / "adr"
    numbers: dict[str, list[str]] = {}
    for p in adr_dir.glob("*.md"):
        name = p.name
        if "HISTORICAL" in name.upper():
            continue
        # leading NNNN-
        parts = name.split("-", 1)
        if parts and parts[0].isdigit():
            numbers.setdefault(parts[0], []).append(name)
    dupes = {k: v for k, v in numbers.items() if len(v) > 1}
    assert not dupes, f"Duplicate ADR numbers: {dupes}"
