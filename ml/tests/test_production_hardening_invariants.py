"""Production-hardening invariants (WS9) and decision-score policy (WS6).

Covers:
- apply_reliability_weight default True
- risk_aversion default 0
- continuous weight bounds and monotonicity
- SHADOW != ACTIVE
- promotion gate fail-closed / canary
- invalid reliability_weight_mode
- decision score canonical behaviour
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _ml_db_url(monkeypatch):
    monkeypatch.setenv("ML_DATABASE_URL", "postgresql://x:x@localhost/x")


# ── Decision score ──────────────────────────────────────────────────────────

def test_decision_score_defaults_apply_weight():
    from ml.auction.decision_score import compute_decision_score

    # weight 0.5 → score halved when apply=True (default)
    assert compute_decision_score(
        projected_score=10.0, reliability_weight=0.5
    ) == pytest.approx(5.0)

    # apply=False ignores weight
    assert compute_decision_score(
        projected_score=10.0, reliability_weight=0.5, apply_reliability_weight=False
    ) == pytest.approx(10.0)


def test_decision_score_risk_aversion():
    from ml.auction.decision_score import compute_decision_score

    assert compute_decision_score(
        projected_score=10.0,
        prediction_std=2.0,
        risk_aversion=0.5,
        apply_reliability_weight=False,
    ) == pytest.approx(9.0)


def test_decision_score_from_player_dict():
    from ml.auction.decision_score import compute_decision_score_from_player

    p = {
        "projected_score": 8.0,
        "reliability_weight": 0.5,
        "prediction_std": 1.0,
    }
    assert compute_decision_score_from_player(
        p, apply_reliability_weight=True, risk_aversion=0.0
    ) == pytest.approx(4.0)


def test_adzic_style_limited_decision_score_below_raw():
    """Golden: minutes=163 LIMITED player must be discounted when weight applied."""
    from ml.auction.decision_score import compute_decision_score
    from ml.sample_reliability import continuous_reliability_weight

    minutes = 163
    rw = continuous_reliability_weight(minutes)
    assert 0.3 <= rw < 1.0
    raw = 12.0
    decision = compute_decision_score(
        projected_score=raw, reliability_weight=rw, apply_reliability_weight=True
    )
    assert decision < raw
    assert decision == pytest.approx(raw * rw)


# ── Config defaults ─────────────────────────────────────────────────────────

def test_auction_config_default_apply_true():
    from ml.auction.models import AuctionConfig

    # Minimal required fields — inspect field default
    field = AuctionConfig.__dataclass_fields__["apply_reliability_weight"]
    assert field.default is True


def test_schema_default_apply_true():
    from api.src.schemas import AuctionConfigSchema

    # Pydantic model default
    defaults = AuctionConfigSchema.model_fields
    assert defaults["apply_reliability_weight"].default is True
    assert defaults["risk_aversion"].default == 0.0


def test_mlconfig_rejects_invalid_mode():
    from pydantic import ValidationError
    from ml.config import MLConfig

    with pytest.raises(ValidationError):
        MLConfig(database_url="postgresql://x:x@localhost/x", reliability_weight_mode="foo")
    with pytest.raises(ValidationError):
        MLConfig(database_url="postgresql://x:x@localhost/x", reliability_weight_mode="true")


def test_mlconfig_accepts_bucket_and_continuous():
    from ml.config import MLConfig

    for mode in ("bucket", "continuous", "BUCKET", " Continuous "):
        cfg = MLConfig(
            database_url="postgresql://x:x@localhost/x",
            reliability_weight_mode=mode,
        )
        assert cfg.reliability_weight_mode in ("bucket", "continuous")


# ── Continuous weight invariants ────────────────────────────────────────────

def test_continuous_weight_bounds_and_monotonic():
    from ml.sample_reliability import continuous_reliability_weight

    prev = -1.0
    for m in range(0, 900, 25):
        w = continuous_reliability_weight(m)
        assert 0.0 <= w <= 1.0
        assert w >= prev - 1e-12
        prev = w
    assert continuous_reliability_weight(800) == pytest.approx(1.0)
    assert continuous_reliability_weight(1000) == pytest.approx(1.0)


# ── Rollout SHADOW != ACTIVE ────────────────────────────────────────────────

def test_shadow_does_not_use_challenger():
    from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController

    ctrl = RolloutController(
        flag=FeatureFlag.PER90_SHRINKAGE,
        stage=FlagStage.SHADOW,
        rollout_pct=100.0,
        random_seed=42,
    )
    assert ctrl.is_active() is False
    assert ctrl.use_challenger() is False


def test_active_can_use_challenger():
    from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController

    ctrl = RolloutController(
        flag=FeatureFlag.PER90_SHRINKAGE,
        stage=FlagStage.ACTIVE,
        rollout_pct=100.0,
        random_seed=42,
    )
    assert ctrl.is_active() is True
    assert ctrl.use_challenger() is True


# ── Promotion gate ──────────────────────────────────────────────────────────

def _write_report(tmp_path: Path, variant_payload: dict) -> Path:
    report = {
        "variants": {
            "A_control": {
                "status": "ok",
                "mae": 1.0,
                "rmse": 1.2,
                "mae_by_cohort": {"STANDARD": 0.9, "LIMITED": 1.1},
                "rmse_by_cohort": {"STANDARD": 1.1, "LIMITED": 1.3},
                "phenom_leakage_rate": 0.05,
                "canary_anomalies_remaining": 0,
            },
            "C_shrinkage": variant_payload,
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_promotion_gate_pass(tmp_path):
    from ml.scripts.check_promotion_gate import main

    payload = {
        "status": "ok",
        "mae": 1.01,
        "rmse": 1.22,
        "mae_by_cohort": {"STANDARD": 0.91, "LIMITED": 1.15},
        "rmse_by_cohort": {"STANDARD": 1.12, "LIMITED": 1.35},
        "phenom_leakage_rate": 0.08,
        "canary_anomalies_remaining": 0,
        "overrepresentation_delta": 0.02,
    }
    path = _write_report(tmp_path, payload)
    assert main([str(path)]) == 0


def test_promotion_gate_canary_remaining_denies(tmp_path):
    from ml.scripts.check_promotion_gate import main

    payload = {
        "status": "ok",
        "mae": 1.0,
        "rmse": 1.2,
        "mae_by_cohort": {"STANDARD": 0.9, "LIMITED": 1.0},
        "rmse_by_cohort": {"STANDARD": 1.1, "LIMITED": 1.2},
        "phenom_leakage_rate": 0.05,
        "canary_anomalies_remaining": 1,
    }
    path = _write_report(tmp_path, payload)
    assert main([str(path)]) == 1


def test_promotion_gate_missing_canary_denies(tmp_path):
    from ml.scripts.check_promotion_gate import main

    payload = {
        "status": "ok",
        "mae": 1.0,
        "rmse": 1.2,
        "mae_by_cohort": {"STANDARD": 0.9, "LIMITED": 1.0},
        "rmse_by_cohort": {"STANDARD": 1.1, "LIMITED": 1.2},
        "phenom_leakage_rate": 0.05,
        # canary_anomalies_remaining intentionally omitted
    }
    path = _write_report(tmp_path, payload)
    assert main([str(path)]) == 1


def test_promotion_gate_missing_report_exit_2(tmp_path):
    from ml.scripts.check_promotion_gate import main

    assert main([str(tmp_path / "nope.json")]) == 2


# ── Workflow SHADOW mapping (static) ────────────────────────────────────────

def test_workflow_does_not_treat_shadow_as_active():
    yaml = (REPO_ROOT / ".github/workflows/ml-training.yml").read_text(encoding="utf-8")
    # Old antipattern must be gone
    assert 'stage == "shadow" or .value.stage == "active"' not in yaml
    assert "ENABLED_FLAGS" not in yaml or "ACTIVE_FLAGS" in yaml
    assert "CHALLENGER" in yaml
    assert "ACTIVE_FLAGS" in yaml
    assert "SHADOW_FLAGS" in yaml
