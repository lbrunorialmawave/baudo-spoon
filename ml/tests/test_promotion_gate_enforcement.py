"""Tests for WS6 — promotion gate hard enforcement.

Plan §8 makes it impossible to reach ``ACTIVE`` without a passing
promotion gate.  These tests exercise every branch of
:meth:`RolloutController.promote_to_active` plus the structured
counterpart of ``check_promotion_gate`` that the controller calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.rollout import (
    AuditKind,
    AuditLog,
    FeatureFlag,
    FlagStage,
    PromotionGateDenied,
    PromotionGateError,
    PromotionGateReport,
    RolloutController,
)
from ml.rollout.controller import (
    DEFAULT_ROLLOUT_PCT,
    GateFn,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_report(
    tmp_path: Path,
    *,
    variant_status: str = "ok",
    mae: float = 0.30,
    phenom_leakage_rate: float = 0.10,
    canary_anomalies_remaining: int = 0,
    limited_mae: float = 0.35,
    standard_mae: float = 0.28,
    overrep_delta_pp: float | None = None,
) -> Path:
    """Write a healthy promotion report to ``tmp_path`` and return the path."""
    variant: dict[str, Any] = {
        "status": variant_status,
        "mae": mae,
        "rmse": 0.40,
        "mae_by_cohort": {
            "STANDARD": standard_mae,
            "LIMITED": limited_mae,
            "INSUFFICIENT": None,
        },
        "rmse_by_cohort": {
            "STANDARD": 0.38,
            "LIMITED": 0.45,
            "INSUFFICIENT": None,
        },
        "phenom_leakage_rate": phenom_leakage_rate,
        "canary_anomalies_remaining": canary_anomalies_remaining,
        "canary_anomalies_total": 0,
        "canary_anomalies_resolved": 0,
    }
    if overrep_delta_pp is not None:
        variant["phenom_overrepresentation"] = 1.0
        variant["overrepresentation_delta"] = overrep_delta_pp
    report = {
        "run_id": "test-run",
        "config_hash": None,
        "variants": {
            "A_control": {
                "status": "ok",
                "mae": 0.30,
                "rmse": 0.40,
            },
            "C_shrinkage": variant,
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _passing_gate(report_path: Path, **kwargs: Any) -> PromotionGateReport:
    """A gate function that always PASSES."""
    return PromotionGateReport(
        passed=True,
        failures=(),
        report_path=str(report_path),
        variant="C_shrinkage",
    )


def _failing_gate(
    report_path: Path, *, msg: str = "simulated gate failure", **kwargs: Any
) -> PromotionGateReport:
    """A gate function that always FAILS with a single, recognisable message."""
    return PromotionGateReport(
        passed=False,
        failures=(msg,),
        report_path=str(report_path),
        variant="C_shrinkage",
    )


def _make_controller(
    *,
    stage: FlagStage = FlagStage.SHADOW,
    rollout_pct: float = 100.0,
    audit_log: AuditLog | None = None,
    gate_fn: GateFn | None = None,
) -> RolloutController:
    return RolloutController(
        flag=FeatureFlag.PER90_SHRINKAGE,
        stage=stage,
        rollout_pct=rollout_pct,
        audit_log=audit_log,
        gate_fn=gate_fn,
    )


# ── PromotionGateReport ──────────────────────────────────────────────────


class TestPromotionGateReport:
    def test_to_dict_normalises_tuple_to_list(self) -> None:
        r = PromotionGateReport(
            passed=False,
            failures=("a", "b"),
            report_path="/x",
            variant="v",
        )
        d = r.to_dict()
        assert d["failures"] == ["a", "b"]
        assert d["passed"] is False
        assert d["report_path"] == "/x"
        assert d["variant"] == "v"

    def test_default_fields(self) -> None:
        r = PromotionGateReport(
            passed=True,
            failures=(),
            report_path="/x",
            variant="v",
        )
        assert r.control == "A_control"
        assert r.config_hash is None
        assert r.config_hash_status is None
        assert r.config_snapshot_path is None
        assert r.extra == {}


# ── Default controller behaviour ──────────────────────────────────────────


class TestDefaultGateFn:
    def test_default_gate_fn_is_evaluate_report(self) -> None:
        c = _make_controller()
        # The default gate function must be the canonical evaluate_report.
        from ml.scripts.check_promotion_gate import evaluate_report

        assert c.gate_fn is evaluate_report

    def test_promote_to_active_passes_with_healthy_report(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller()
        outcome = c.promote_to_active(
            report_path=report,
            actor="ci-bot",
            commit_sha="abc1234",
        )
        assert outcome.passed is True
        assert c.stage == FlagStage.ACTIVE
        assert c.rollout_pct == DEFAULT_ROLLOUT_PCT

    def test_promote_to_active_event_records_gate_pass(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller()
        c.promote_to_active(
            report_path=report,
            actor="ci-bot",
            commit_sha="abc1234",
        )
        last = c.events[-1]
        assert last["to_stage"] == "active"
        assert last["gate_result"] == "PASS"
        assert last["actor"] == "ci-bot"
        assert last["commit_sha"] == "abc1234"
        assert last["report_path"] == str(report)
        assert last["from_stage"] == "shadow"


# ── Gate failure blocks transition ────────────────────────────────────────


class TestGateFailureBlocksTransition:
    def test_promote_to_active_denied_when_gate_fails(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        with pytest.raises(PromotionGateDenied) as excinfo:
            c.promote_to_active(report_path=report, actor="ci-bot")
        # Stage does NOT change.
        assert c.stage == FlagStage.SHADOW
        assert c.rollout_pct == 100.0
        # Outcome is exposed on the exception.
        assert excinfo.value.outcome.passed is False
        assert "simulated gate failure" in excinfo.value.outcome.failures

    def test_promote_to_active_denied_on_canary_anomalies(self, tmp_path: Path) -> None:
        # Use the real default gate: anomalies_remaining=1 must deny.
        report = _make_report(tmp_path, canary_anomalies_remaining=1)
        c = _make_controller()
        with pytest.raises(PromotionGateDenied) as excinfo:
            c.promote_to_active(report_path=report, actor="ci-bot")
        assert c.stage == FlagStage.SHADOW
        assert any("canary_anomalies_remaining" in f for f in excinfo.value.outcome.failures)

    def test_promote_to_active_denied_on_high_phenom_leakage(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path, phenom_leakage_rate=0.80)
        c = _make_controller()
        with pytest.raises(PromotionGateDenied):
            c.promote_to_active(report_path=report, actor="ci-bot")
        assert c.stage == FlagStage.SHADOW

    def test_promote_to_active_denied_when_report_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.json"
        c = _make_controller()
        with pytest.raises(PromotionGateDenied) as excinfo:
            c.promote_to_active(report_path=missing, actor="ci-bot")
        assert c.stage == FlagStage.SHADOW
        assert any("evaluation error" in f for f in excinfo.value.outcome.failures)

    def test_promote_to_active_denied_on_severe_overrepresentation(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(tmp_path, overrep_delta_pp=15.0)
        c = _make_controller()
        with pytest.raises(PromotionGateDenied):
            c.promote_to_active(report_path=report, actor="ci-bot")
        assert c.stage == FlagStage.SHADOW


# ── Audit log integration ─────────────────────────────────────────────────


class TestAuditLogIntegration:
    def test_promote_to_active_records_passed_transition(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        log = AuditLog()
        c = _make_controller(audit_log=log)
        c.promote_to_active(report_path=report, actor="ci-bot", commit_sha="abc")
        assert len(log) == 1
        rec = log.records[0]
        assert rec.kind == AuditKind.TRANSITION
        assert rec.flag == "enable_shrinkage"
        assert rec.from_stage == "shadow"
        assert rec.to_stage == "active"
        assert rec.reason == "promotion_gate_passed"
        assert rec.gate_result == "PASS"
        assert rec.promotion_report == str(report)
        assert rec.actor == "ci-bot"
        assert rec.commit_sha == "abc"

    def test_promote_to_active_records_denied_when_gate_fails(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        log = AuditLog()
        c = _make_controller(audit_log=log, gate_fn=_failing_gate)
        with pytest.raises(PromotionGateDenied):
            c.promote_to_active(report_path=report, actor="ci-bot")
        assert len(log) == 1
        rec = log.records[0]
        assert rec.kind == AuditKind.DENIED
        assert rec.reason == "promotion_gate_failed"
        assert "simulated gate failure" in rec.failed_checks
        assert rec.flag == "enable_shrinkage"
        assert rec.from_stage == "shadow"
        assert rec.to_stage == "active"

    def test_no_audit_log_means_no_audit_record(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller()  # no audit_log
        c.promote_to_active(report_path=report, actor="ci-bot")
        # No exception — events still recorded, but no audit log to write to.
        assert c.stage == FlagStage.ACTIVE


# ── Break-glass (WS6 §8.5) ────────────────────────────────────────────────


class TestBreakGlass:
    def test_break_glass_overrides_gate_failure(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        log = AuditLog()
        c = _make_controller(audit_log=log, gate_fn=_failing_gate)
        outcome = c.promote_to_active(
            report_path=report,
            actor="oncall",
            break_glass=True,
            break_glass_reason="incident INC-42 — need to flip the flag NOW",
        )
        # Transition happened despite the gate failure.
        assert c.stage == FlagStage.ACTIVE
        assert outcome.passed is False
        # Audit log carries the override.
        assert len(log) == 1
        rec = log.records[0]
        assert rec.kind == AuditKind.TRANSITION
        assert rec.reason == "break_glass"
        assert rec.gate_result == "BREAK_GLASS"
        assert "incident INC-42" in (rec.extra or {}).get("break_glass_reason", "")

    def test_break_glass_requires_non_empty_reason(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        with pytest.raises(ValueError, match="break_glass_reason"):
            c.promote_to_active(
                report_path=report,
                actor="oncall",
                break_glass=True,
                break_glass_reason="",
            )
        assert c.stage == FlagStage.SHADOW

    def test_break_glass_ignores_blank_string_reason(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        with pytest.raises(ValueError, match="break_glass_reason"):
            c.promote_to_active(
                report_path=report,
                actor="oncall",
                break_glass=True,
                break_glass_reason="   ",
            )
        assert c.stage == FlagStage.SHADOW

    def test_break_glass_with_passing_gate_acts_as_normal_promotion(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(tmp_path)
        log = AuditLog()
        c = _make_controller(audit_log=log, gate_fn=_passing_gate)
        c.promote_to_active(
            report_path=report,
            actor="oncall",
            break_glass=True,
            break_glass_reason="non-emergency — should not normally happen",
        )
        # Transition succeeded; reason is still break_glass because the
        # operator set the flag, but gate_result reflects the actual gate.
        assert c.stage == FlagStage.ACTIVE
        rec = log.records[0]
        assert rec.gate_result == "BREAK_GLASS"
        assert rec.reason == "break_glass"

    def test_break_glass_event_includes_failures(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        c.promote_to_active(
            report_path=report,
            actor="oncall",
            break_glass=True,
            break_glass_reason="INC-42",
        )
        last = c.events[-1]
        assert last["reason"] == "break_glass"
        assert last["break_glass_reason"] == "INC-42"
        assert "simulated gate failure" in last["gate_failures"]


# ── Custom gate_fn injection ─────────────────────────────────────────────


class TestCustomGateFn:
    def test_custom_gate_fn_called_with_report_path(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        calls: list[Path] = []

        def custom_gate(report_path: Path, **kwargs: Any) -> PromotionGateReport:
            calls.append(report_path)
            return _passing_gate(report_path)

        c = _make_controller(gate_fn=custom_gate)
        c.promote_to_active(report_path=report, actor="ci-bot")
        assert calls == [report]

    def test_custom_gate_fn_receives_config_snapshot_path(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(tmp_path)
        snap = tmp_path / "snap.json"
        snap.write_text("{}", encoding="utf-8")
        received: dict[str, Any] = {}

        def custom_gate(report_path: Path, **kwargs: Any) -> PromotionGateReport:
            received.update(kwargs)
            return _passing_gate(report_path)

        c = _make_controller(gate_fn=custom_gate)
        c.promote_to_active(
            report_path=report,
            config_snapshot_path=snap,
            actor="ci-bot",
        )
        assert received["config_snapshot"] == snap

    def test_failing_custom_gate_blocks_promotion(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        with pytest.raises(PromotionGateDenied):
            c.promote_to_active(report_path=report, actor="ci-bot")
        assert c.stage == FlagStage.SHADOW


# ── Non-ACTIVE transitions do not require the gate ───────────────────────


class TestNonActiveTransitionsUnaffected:
    def test_promote_to_shadow_does_not_invoke_gate(self, tmp_path: Path) -> None:
        c = _make_controller(
            stage=FlagStage.DISABLED,
            rollout_pct=0.0,
            gate_fn=_failing_gate,
        )
        # promote() must still work for non-ACTIVE transitions; the gate
        # is only consulted by promote_to_active().
        c.promote(new_stage=FlagStage.SHADOW)
        assert c.stage == FlagStage.SHADOW
        # The failing gate is never called — promote() does not consult it.

    def test_promote_method_legacy_contract_intact(self) -> None:
        # promote() must still work atomically without consulting a gate.
        c = _make_controller(stage=FlagStage.SHADOW, gate_fn=_failing_gate)
        c.promote(new_stage=FlagStage.ACTIVE, new_rollout_pct=50.0)
        assert c.stage == FlagStage.ACTIVE
        # The failing gate is never consulted for non-gate-protected calls.


# ── Config hash integration ───────────────────────────────────────────────


class TestConfigHashIntegration:
    def test_audit_record_includes_config_hash_when_snapshot_provided(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(tmp_path)
        snap_map = {
            "enable_limited_sample_training": False,
            "enable_shrinkage": True,
            "reliability_weight_mode": "bucket",
        }
        log = AuditLog()
        c = _make_controller(audit_log=log)
        c.promote_to_active(
            report_path=report,
            config_snapshot=snap_map,
            actor="ci-bot",
        )
        rec = log.records[0]
        assert rec.config_hash is not None
        assert rec.config_hash.startswith("sha256:")

    def test_break_glass_audit_record_carries_config_hash(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(tmp_path)
        snap_map = {"enable_shrinkage": True}
        log = AuditLog()
        c = _make_controller(audit_log=log, gate_fn=_failing_gate)
        c.promote_to_active(
            report_path=report,
            config_snapshot=snap_map,
            actor="oncall",
            break_glass=True,
            break_glass_reason="INC-42",
        )
        rec = log.records[0]
        assert rec.config_hash is not None


# ── PromotionGateError base class ─────────────────────────────────────────


class TestPromotionGateError:
    def test_denied_is_subclass_of_error(self) -> None:
        assert issubclass(PromotionGateDenied, PromotionGateError)
        assert issubclass(PromotionGateError, Exception)

    def test_denied_carries_outcome_attribute(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller(gate_fn=_failing_gate)
        with pytest.raises(PromotionGateDenied) as excinfo:
            c.promote_to_active(report_path=report, actor="ci-bot")
        outcome = excinfo.value.outcome
        assert outcome.passed is False
        assert outcome.failures  # non-empty


# ── Defensive behaviour ──────────────────────────────────────────────────


class TestDefensiveBehaviour:
    def test_promote_to_active_without_gate_fn_raises(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = RolloutController(
            flag=FeatureFlag.PER90_SHRINKAGE,
            stage=FlagStage.SHADOW,
            rollout_pct=100.0,
            gate_fn=None,
        )
        # __post_init__ fills it in with the default, so the controller
        # still has a real gate function.  To exercise the "no gate"
        # defensive branch we must monkey-patch the attribute to None.
        c.gate_fn = None
        with pytest.raises(PromotionGateError, match="gate_fn is None"):
            c.promote_to_active(report_path=report, actor="ci-bot")

    def test_invalid_rollout_pct_rejected(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        c = _make_controller()
        with pytest.raises(ValueError):
            c.promote_to_active(
                report_path=report,
                actor="ci-bot",
                new_rollout_pct=200.0,
            )

    def test_audit_log_receives_correct_failure_messages(
        self, tmp_path: Path
    ) -> None:
        report = _make_report(
            tmp_path,
            canary_anomalies_remaining=2,
            phenom_leakage_rate=0.30,
        )
        log = AuditLog()
        c = _make_controller(audit_log=log)
        with pytest.raises(PromotionGateDenied):
            c.promote_to_active(report_path=report, actor="ci-bot")
        rec = log.records[0]
        assert rec.kind == AuditKind.DENIED
        joined = " | ".join(rec.failed_checks)
        assert "canary_anomalies_remaining" in joined
        assert "phenom_leakage_rate" in joined


# ── Idempotency guard (WS16, plan §18) ────────────────────────────────────
#
# Re-running the rollout workflow against an already-ACTIVE flag with
# the same canonical config_hash MUST be a no-op.  This is the second
# half of the fix for Run "Idempotenza" (the first half is the shared
# snapshot in :mod:`ml.rollout.config_snapshot`).


class TestIdempotencyGuard:
    """``promote_to_active`` must short-circuit on re-runs of the same
    effective configuration so a second ``ml-training.yml`` execution
    cannot be denied by a stale report downloaded from R2."""

    @staticmethod
    def _activate(
        controller: RolloutController,
        report: Path,
        snapshot: dict[str, Any],
    ) -> PromotionGateReport:
        return controller.promote_to_active(
            report_path=report,
            config_snapshot=snapshot,
            actor="ci-bot",
        )

    def test_active_with_same_hash_is_noop(
        self, tmp_path: Path
    ) -> None:
        snapshot = {
            "min_minutes": 600,
            "min_minutes_hard": 270,
            "enable_shrinkage": True,
            "reliability_weight_mode": "continuous",
        }
        report = _make_report(tmp_path)
        c = _make_controller(
            stage=FlagStage.SHADOW, gate_fn=_passing_gate
        )
        # First call: real promotion.
        outcome = self._activate(c, report, snapshot)
        assert outcome.passed is True
        assert c.stage == FlagStage.ACTIVE
        first_event_count = len(c.events)
        first_audit_count = (
            len(c.audit_log.records) if c.audit_log is not None else 0
        )

        # Second call: same hash → idempotent replay.  The gate MUST
        # NOT be invoked (we'd notice because the synthetic gate would
        # still PASS, but the noop path records the result without
        # touching ``events`` or ``audit_log``).
        outcome2 = self._activate(c, report, snapshot)
        assert outcome2.passed is True
        assert outcome2.extra.get("idempotent_replay") is True
        assert c.stage == FlagStage.ACTIVE
        # No new transition recorded.
        assert len(c.events) == first_event_count
        if c.audit_log is not None:
            assert len(c.audit_log.records) == first_audit_count

    def test_active_with_different_hash_falls_through_to_gate(
        self, tmp_path: Path
    ) -> None:
        snapshot_a = {
            "min_minutes": 600,
            "min_minutes_hard": 270,
            "enable_shrinkage": True,
        }
        snapshot_b = {**snapshot_a, "min_minutes": 700}  # drift
        report = _make_report(tmp_path)
        # The gate is FAILING on purpose; we want to see the guard
        # short-circuit and the gate take over instead.
        c = _make_controller(
            stage=FlagStage.SHADOW, gate_fn=_failing_gate
        )
        # First call: stage transition is needed (SHADOW → ACTIVE),
        # so the guard does NOT apply yet.
        from ml.rollout import PromotionGateDenied

        with pytest.raises(PromotionGateDenied):
            self._activate(c, report, snapshot_a)
        # Stage was NOT advanced because the gate failed.
        assert c.stage == FlagStage.SHADOW

        # Manually move to ACTIVE so we can exercise the guard.
        c.stage = FlagStage.ACTIVE
        c.rollout_pct = 10.0
        c.promote(
            new_stage=FlagStage.ACTIVE,
            new_rollout_pct=10.0,
            config_snapshot=snapshot_a,
        )
        # Now a re-run with a DIFFERENT hash must NOT short-circuit:
        # the gate is invoked, fails, and we observe a denial.
        with pytest.raises(PromotionGateDenied):
            self._activate(c, report, snapshot_b)

    def test_break_glass_bypasses_idempotency_guard(
        self, tmp_path: Path
    ) -> None:
        snapshot = {
            "min_minutes": 600,
            "min_minutes_hard": 270,
            "enable_shrinkage": True,
        }
        report = _make_report(tmp_path)
        # The gate is FAILING; we want the break-glass path to take
        # over (operator override) without the idempotency short-circuit
        # silently swallowing the failure.
        c = _make_controller(
            stage=FlagStage.SHADOW, gate_fn=_failing_gate
        )
        c.promote(
            new_stage=FlagStage.ACTIVE,
            new_rollout_pct=10.0,
            config_snapshot=snapshot,
        )
        # Re-activate with the same hash but break_glass=True →
        # gate IS invoked and the operator override proceeds.
        outcome = c.promote_to_active(
            report_path=report,
            config_snapshot=snapshot,
            actor="on-call",
            break_glass=True,
            break_glass_reason="incident IR-2026-08-15",
        )
        assert outcome.passed is False  # gate still fails
        # But the break-glass override was applied (event reason).
        last = c.events[-1]
        assert last["reason"] == "break_glass"
        assert last["gate_result"] == "BREAK_GLASS"


# ── Regression: bundle-shaped effective_config.json (Run "Idempotenza") ───
#
# ``run_pipeline._build_effective_config_payload`` persists
# ``effective_config.json`` as a *bundle* — ``{"config":..., "extra":...,
# "config_hash":...}`` — via ``build_config_bundle``. The report's own
# ``config_hash`` (also produced by ``build_config_bundle``) is computed
# over the *bare* inner config only. Before the fix, ``evaluate_report``
# hashed the whole bundle it read from disk, so a config-hash mismatch
# was reported on every single run, even with zero drift. This is
# exactly the "Phase 10 — ACTIVE" CI failure from run
# https://github.com/lbrunorialmawave/baudo-spoon/actions/runs/31898300912.


class TestConfigSnapshotBundleUnwrapping:
    """The real artefact shape (bundle) must hash-match the report."""

    def test_evaluate_report_accepts_bundle_shaped_effective_config(
        self, tmp_path: Path
    ) -> None:
        from ml.rollout.config_hash import build_config_bundle
        from ml.scripts.check_promotion_gate import evaluate_report

        bare_config = {
            "min_minutes": 800,
            "min_minutes_hard": 100,
            "enable_limited_sample_training": False,
            "enable_shrinkage": True,
            "enable_recent_role_features": False,
            "enable_breakout_model": False,
            "weighting_strategy": "sqrt",
            "shrinkage_prior_strength": 300,
            "reliability_weight_mode": "bucket",
        }

        # report.json: config_hash computed over the BARE config, as
        # ``_build_promotion_report_payload`` does in run_pipeline.py.
        report_bundle = build_config_bundle(config=bare_config)
        report = _make_report(tmp_path)
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        report_payload["config_hash"] = report_bundle["config_hash"]
        report.write_text(json.dumps(report_payload), encoding="utf-8")

        # effective_config.json: the real on-disk shape is the FULL
        # bundle, not the bare config — this is what run_pipeline.py
        # actually writes and what ml-training.yml passes as
        # --config-snapshot.
        effective_config_path = tmp_path / "effective_config.json"
        effective_config_path.write_text(
            json.dumps(build_config_bundle(config=bare_config)),
            encoding="utf-8",
        )

        outcome = evaluate_report(
            report,
            variant="C_shrinkage",
            config_snapshot=effective_config_path,
        )

        assert outcome.config_hash_status == "match"
        assert not any("config_hash mismatch" in f for f in outcome.failures)
        assert outcome.passed is True

    def test_controller_idempotency_guard_unwraps_bundle_path(
        self, tmp_path: Path
    ) -> None:
        """The idempotency short-circuit must also survive the bundle shape."""
        from ml.rollout.config_hash import build_config_bundle, compute_config_hash

        bare_config = {
            "min_minutes": 600,
            "min_minutes_hard": 270,
            "enable_shrinkage": True,
        }
        bundle_path = tmp_path / "effective_config.json"
        bundle_path.write_text(
            json.dumps(build_config_bundle(config=bare_config)),
            encoding="utf-8",
        )

        report = _make_report(tmp_path)
        c = _make_controller(stage=FlagStage.SHADOW, gate_fn=_passing_gate)

        c.promote(
            new_stage=FlagStage.ACTIVE,
            new_rollout_pct=10.0,
            config_snapshot=bare_config,
        )
        assert c.events[-1]["config_hash"] == compute_config_hash(bare_config)

        first_event_count = len(c.events)
        outcome = c.promote_to_active(
            report_path=report,
            config_snapshot_path=bundle_path,
            actor="ci-bot",
        )
        assert outcome.extra.get("idempotent_replay") is True
        assert len(c.events) == first_event_count



# ── WS14-bis: variant mapping + provenance ─────────────────────────────────


class TestVariantFlagPropagation:
    def test_promote_to_active_propagates_flag_to_gate_fn(self, tmp_path: Path) -> None:
        """controller.promote_to_active must pass flag=self.flag.value."""
        received: dict[str, Any] = {}

        def recording_gate(report_path, **kwargs):  # type: ignore[no-untyped-def]
            received.update(kwargs)
            received["report_path"] = report_path
            return PromotionGateReport(
                passed=True,
                failures=(),
                report_path=str(report_path),
                variant=kwargs.get("flag", "?"),
            )

        report = _make_report(tmp_path)
        c = RolloutController(
            flag=FeatureFlag.LIMITED_SAMPLE_TRAINING,
            stage=FlagStage.SHADOW,
            gate_fn=recording_gate,
        )
        c.promote_to_active(report_path=report, actor="test")
        assert received.get("flag") == "enable_limited_sample_training"


class TestProvenanceAndFlagOverride:
    def test_evaluate_report_flag_overrides_default_variant(self, tmp_path: Path) -> None:
        from ml.scripts.check_promotion_gate import evaluate_report

        # Harness-shaped report with B_weighting (not the default C_shrinkage).
        report_path = tmp_path / "report.json"
        payload = {
            "variants": {
                "A_control": {
                    "status": "ok",
                    "mae": 0.32,
                    "rmse": 0.40,
                    "mae_by_cohort": {"LIMITED": 0.35, "STANDARD": 0.30},
                    "rmse_by_cohort": {"LIMITED": 0.42, "STANDARD": 0.38},
                    "phenom_leakage_rate": 0.05,
                    "canary_anomalies_remaining": 0,
                },
                "B_weighting": {
                    "status": "ok",
                    "mae": 0.30,
                    "rmse": 0.38,
                    "mae_by_cohort": {"LIMITED": 0.33, "STANDARD": 0.28},
                    "rmse_by_cohort": {"LIMITED": 0.40, "STANDARD": 0.36},
                    "phenom_leakage_rate": 0.04,
                    "canary_anomalies_remaining": 0,
                },
            }
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        outcome = evaluate_report(
            report_path,
            flag="enable_limited_sample_training",
        )
        assert outcome.variant == "B_weighting"
        assert outcome.passed is True

    def test_evaluate_report_explicit_variant_wins_over_flag(self, tmp_path: Path) -> None:
        from ml.scripts.check_promotion_gate import evaluate_report

        report_path = tmp_path / "report.json"
        payload = {
            "variants": {
                "A_control": {
                    "status": "ok",
                    "mae": 0.32,
                    "rmse": 0.40,
                    "mae_by_cohort": {"LIMITED": 0.35, "STANDARD": 0.30},
                    "rmse_by_cohort": {"LIMITED": 0.42, "STANDARD": 0.38},
                    "phenom_leakage_rate": 0.05,
                    "canary_anomalies_remaining": 0,
                },
                "C_shrinkage": {
                    "status": "ok",
                    "mae": 0.29,
                    "rmse": 0.37,
                    "mae_by_cohort": {"LIMITED": 0.32, "STANDARD": 0.27},
                    "rmse_by_cohort": {"LIMITED": 0.39, "STANDARD": 0.35},
                    "phenom_leakage_rate": 0.03,
                    "canary_anomalies_remaining": 0,
                },
            }
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        # Explicit variant=C_shrinkage must win even if flag maps to B.
        outcome = evaluate_report(
            report_path,
            variant="C_shrinkage",
            flag="enable_limited_sample_training",
        )
        assert outcome.variant == "C_shrinkage"
        assert outcome.passed is True

    def test_convenience_report_rejected_for_non_control_variant(
        self, tmp_path: Path
    ) -> None:
        from ml.scripts.check_promotion_gate import evaluate_report

        report_path = tmp_path / "convenience.json"
        payload = {
            "variants": {
                "A_control": {
                    "status": "ok",
                    "mae": 0.32,
                    "rmse": 0.40,
                    "mae_by_cohort": {"LIMITED": 0.35, "STANDARD": 0.30},
                    "rmse_by_cohort": {"LIMITED": 0.42, "STANDARD": 0.38},
                    "phenom_leakage_rate": 0.05,
                    "canary_anomalies_remaining": 0,
                }
            }
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        outcome = evaluate_report(report_path, variant="C_shrinkage")
        assert outcome.passed is False
        assert any("convenience report" in f for f in outcome.failures)

    def test_harness_report_with_two_variants_passes_provenance_check(
        self, tmp_path: Path
    ) -> None:
        from ml.scripts.check_promotion_gate import evaluate_report

        report_path = tmp_path / "harness.json"
        payload = {
            "variants": {
                "A_control": {
                    "status": "ok",
                    "mae": 0.32,
                    "rmse": 0.40,
                    "mae_by_cohort": {"LIMITED": 0.35, "STANDARD": 0.30},
                    "rmse_by_cohort": {"LIMITED": 0.42, "STANDARD": 0.38},
                    "phenom_leakage_rate": 0.05,
                    "canary_anomalies_remaining": 0,
                },
                "C_shrinkage": {
                    "status": "ok",
                    "mae": 0.29,
                    "rmse": 0.37,
                    "mae_by_cohort": {"LIMITED": 0.32, "STANDARD": 0.27},
                    "rmse_by_cohort": {"LIMITED": 0.39, "STANDARD": 0.35},
                    "phenom_leakage_rate": 0.03,
                    "canary_anomalies_remaining": 0,
                },
            }
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        outcome = evaluate_report(report_path, variant="C_shrinkage")
        assert outcome.passed is True
        assert not any("convenience" in f for f in outcome.failures)

    def test_e2e_shadow_flag_cannot_promote_without_harness_report(
        self, tmp_path: Path
    ) -> None:
        """Regression: convenience report + flag still in SHADOW must DENY."""
        from ml.scripts.check_promotion_gate import evaluate_report

        # Simulate the convenience report that run_pipeline would emit
        # when enable_shrinkage is still False (SHADOW).
        report_path = tmp_path / "promotion_report.json"
        payload = {
            "variants": {
                "A_control": {
                    "status": "ok",
                    "mae": 0.32,
                    "rmse": 0.40,
                    "mae_by_cohort": {"LIMITED": 0.35, "STANDARD": 0.30},
                    "rmse_by_cohort": {"LIMITED": 0.42, "STANDARD": 0.38},
                    "phenom_leakage_rate": 0.05,
                    "canary_anomalies_remaining": 0,
                }
            }
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        outcome = evaluate_report(
            report_path,
            flag="enable_shrinkage",
        )
        assert outcome.passed is False
        assert outcome.variant == "C_shrinkage"
        assert any(
            "convenience report" in f or "harness report" in f
            for f in outcome.failures
        )
