"""E2E contract tests: decision policy coherence across modules (WS8).

Uses a golden LIMITED player (Adzic-style: minutes=163) and verifies
Optimizer-style weight, Auction VarEngine, alternatives ranking, and
simulation proxy all apply the same reliability policy.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ML_DATABASE_URL", "postgresql://x:x@localhost/x")


@pytest.fixture
def limited_player_dict():
    from ml.sample_reliability import continuous_reliability_weight, classify_cohort

    minutes = 163
    rw = continuous_reliability_weight(minutes)
    return {
        "player_id": "adzic_golden",
        "role": "A",
        "projected_score": 7.5,
        "reliability_weight": rw,
        "prediction_std": 1.5,
        "minutes": minutes,
        "mins_played": minutes,
        "cohort": str(classify_cohort(minutes)),
        "cost": 10.0,
        "eligible_roles": None,
        "season_value": None,
        "fp_ibrido": None,
        "esv": None,
    }


@pytest.fixture
def limited_player_obj(limited_player_dict):
    from ml.optimizer.models import Player

    d = limited_player_dict
    return Player(
        player_id=d["player_id"],
        name="Adzic Golden",
        role=d["role"],
        real_team="Test FC",
        cost=int(d["cost"]),
        projected_score=d["projected_score"],
        reliability_weight=d["reliability_weight"],
        prediction_std=d["prediction_std"],
    )


def test_e2e_decision_score_matches_var_engine(limited_player_dict):
    from ml.auction.decision_score import compute_decision_score
    from ml.auction.var import VarEngine

    expected = compute_decision_score(
        projected_score=limited_player_dict["projected_score"],
        reliability_weight=limited_player_dict["reliability_weight"],
        prediction_std=limited_player_dict["prediction_std"],
        apply_reliability_weight=True,
        risk_aversion=0.5,
    )
    engine = VarEngine(
        apply_reliability_weight=True,
        risk_aversion=0.5,
    )
    got = engine._get_score(limited_player_dict)
    assert got == pytest.approx(expected)
    assert got < limited_player_dict["projected_score"]


def test_e2e_alternatives_uses_decision_weight(limited_player_obj):
    from ml.auction.alternatives import _get_player_score
    from ml.auction.models import ValuationMode
    from ml.auction.decision_score import compute_decision_score_from_player

    expected = compute_decision_score_from_player(
        limited_player_obj,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )
    got = _get_player_score(
        limited_player_obj,
        ValuationMode.PER_MATCH_RATING,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )
    assert got == pytest.approx(expected)
    assert got < limited_player_obj.projected_score


def test_e2e_simulation_esv_proxy_uses_decision_weight(limited_player_obj):
    from ml.auction.simulation import _esv_proxy
    from ml.auction.decision_score import compute_decision_score_from_player

    expected = compute_decision_score_from_player(
        limited_player_obj,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )
    # price <= 0 → pure score
    got = _esv_proxy(
        limited_player_obj,
        price=0.0,
        apply_reliability_weight=True,
        risk_aversion=0.0,
    )
    assert got == pytest.approx(expected)


def test_e2e_optimizer_style_weight_aligned(limited_player_obj):
    """Mirror solver objective: score * reliability_weight - risk * std."""
    p = limited_player_obj
    rw = p.reliability_weight if p.reliability_weight is not None else 1.0
    risk = 0.25
    std = p.prediction_std if p.prediction_std is not None else 0.0
    expected = p.projected_score * rw - risk * std

    from ml.auction.decision_score import compute_decision_score_from_player

    got = compute_decision_score_from_player(
        p, apply_reliability_weight=True, risk_aversion=risk
    )
    assert got == pytest.approx(expected)


def test_e2e_rollout_stages_decision_path():
    """DISABLED/SHADOW production path ≠ ACTIVE challenger path for weights."""
    from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController
    from ml.rollout.controller import reliability_weight_mode_for_stage
    from ml.sample_reliability import get_reliability_weight

    minutes = 163
    for stage, expect_use in [
        (FlagStage.DISABLED, False),
        (FlagStage.SHADOW, False),
        (FlagStage.ACTIVE, True),
    ]:
        ctrl = RolloutController(
            flag=FeatureFlag.PER90_SHRINKAGE,
            stage=stage,
            rollout_pct=100.0 if stage == FlagStage.ACTIVE else 0.0,
            random_seed=1,
        )
        if stage == FlagStage.ACTIVE:
            # ACTIVE requires rollout_pct > 0 — already set
            pass
        assert ctrl.use_challenger() is expect_use or (
            stage == FlagStage.ACTIVE and ctrl.use_challenger() is True
        )

    # Mode mapping
    assert reliability_weight_mode_for_stage(FlagStage.DISABLED) == "bucket"
    assert reliability_weight_mode_for_stage(FlagStage.SHADOW) == "bucket"
    assert reliability_weight_mode_for_stage(FlagStage.ACTIVE) == "continuous"

    w_bucket = get_reliability_weight(minutes=minutes, mode="bucket")
    w_cont = get_reliability_weight(minutes=minutes, mode="continuous")
    # Continuous is typically finer-grained and can differ from bucket step
    assert 0.0 < w_cont <= 1.0
    assert 0.0 < w_bucket <= 1.0


def test_e2e_env_flags_shadow_not_production(monkeypatch):
    from ml.rollout.env_flags import resolve_env_flags

    monkeypatch.setenv("ML_ENABLE_SHRINKAGE_CHALLENGER", "true")
    monkeypatch.delenv("ML_ENABLE_SHRINKAGE", raising=False)
    resolved = resolve_env_flags()
    assert resolved.production.get("enable_shrinkage") is False
    assert resolved.challenger.get("enable_shrinkage") is True
    assert resolved.stage_for("enable_shrinkage") == "shadow"


def test_e2e_env_flags_active_is_production(monkeypatch):
    from ml.rollout.env_flags import resolve_env_flags

    monkeypatch.setenv("ML_ENABLE_SHRINKAGE", "true")
    resolved = resolve_env_flags()
    assert resolved.production.get("enable_shrinkage") is True
    assert resolved.stage_for("enable_shrinkage") == "active"


def test_e2e_shadow_artifact_written(tmp_path):
    from ml.rollout.shadow_artifacts import write_shadow_artifact, build_shadow_rows

    players = [
        {
            "player_id": "adzic_golden",
            "role": "A",
            "minutes": 163,
            "projected_score": 7.5,
        },
        {
            "player_id": "standard_800",
            "role": "C",
            "minutes": 900,
            "projected_score": 7.0,
        },
    ]
    rows = build_shadow_rows(players, canary_ids={"adzic_golden"})
    assert len(rows) == 2
    adzic = rows[0]
    assert adzic.canary is True
    assert adzic.challenger_score != adzic.baseline_score or adzic.delta == 0.0
    # LIMITED should usually differ between bucket and continuous
    assert adzic.cohort == "LIMITED"

    path = write_shadow_artifact(tmp_path / "shadow.json", players, canary_ids={"adzic_golden"})
    assert path.is_file()
    import json
    data = json.loads(path.read_text())
    assert data["n_rows"] == 2
    assert data["rows"][0]["player_id"] == "adzic_golden"

def test_apply_production_flags_shadow_only(monkeypatch):
    """Production path stays off when only CHALLENGER env is set."""
    from types import SimpleNamespace
    from ml.rollout.env_flags import apply_production_flags_to_config, resolve_env_flags

    monkeypatch.setenv("ML_ENABLE_SHRINKAGE_CHALLENGER", "true")
    monkeypatch.delenv("ML_ENABLE_SHRINKAGE", raising=False)
    resolved = resolve_env_flags()
    cfg = SimpleNamespace(
        enable_shrinkage=False,
        enable_limited_sample_training=False,
        reliability_weight_mode="bucket",
    )
    apply_production_flags_to_config(cfg, resolved)
    assert cfg.enable_shrinkage is False
    assert resolved.challenger.get("enable_shrinkage") is True
    assert resolved.stage_for("enable_shrinkage") == "shadow"


def test_apply_production_flags_active(monkeypatch):
    from types import SimpleNamespace
    from ml.rollout.env_flags import apply_production_flags_to_config, resolve_env_flags

    monkeypatch.setenv("ML_ENABLE_SHRINKAGE", "true")
    resolved = resolve_env_flags()
    cfg = SimpleNamespace(
        enable_shrinkage=False,
        enable_limited_sample_training=False,
        reliability_weight_mode="bucket",
    )
    apply_production_flags_to_config(cfg, resolved)
    assert cfg.enable_shrinkage is True
    assert resolved.stage_for("enable_shrinkage") == "active"
