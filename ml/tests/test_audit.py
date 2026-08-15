"""Tests for ml.rollout.audit (WS15 of plan.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.rollout.audit import (
    AuditKind,
    AuditLog,
    AuditRecord,
    read_audit_log,
    record_denied,
    record_transition,
    records_from_controller_events,
    write_audit_log,
)
from ml.rollout.controller import FeatureFlag, FlagStage, RolloutController


# ── Builders ──────────────────────────────────────────────────────────────


class TestRecordTransition:
    def test_minimal_args(self) -> None:
        r = record_transition(
            actor="ci",
            flag="enable_shrinkage",
            from_stage="shadow",
            to_stage="active",
            from_pct=10.0,
            to_pct=25.0,
            reason="promotion",
        )
        assert r.kind == AuditKind.TRANSITION
        assert r.actor == "ci"
        assert r.flag == "enable_shrinkage"
        assert r.from_stage == "shadow"
        assert r.to_stage == "active"
        assert r.from_pct == 10.0
        assert r.to_pct == 25.0
        assert r.reason == "promotion"
        assert r.gate_result is None
        assert r.config_hash is None
        assert r.commit_sha is None
        assert r.timestamp  # auto-generated

    def test_full_args(self) -> None:
        r = record_transition(
            actor="ci",
            flag="enable_shrinkage",
            from_stage="shadow",
            to_stage="active",
            from_pct=10.0,
            to_pct=25.0,
            reason="promotion",
            commit_sha="abc1234",
            promotion_report="ml/reports/report.json",
            gate_result="PASS",
            config_hash="sha256:deadbeef",
            extra={"cohort_mae": {"LIMITED": 0.42}},
            timestamp="2026-08-15T11:00:00+00:00",
        )
        d = r.to_dict()
        assert d["commit_sha"] == "abc1234"
        assert d["promotion_report"] == "ml/reports/report.json"
        assert d["gate_result"] == "PASS"
        assert d["config_hash"] == "sha256:deadbeef"
        assert d["extra"]["cohort_mae"]["LIMITED"] == 0.42
        assert d["timestamp"] == "2026-08-15T11:00:00+00:00"
        assert d["kind"] == "transition"

    def test_to_dict_serialisable(self) -> None:
        r = record_transition(
            actor="ci",
            flag="x",
            from_stage="disabled",
            to_stage="shadow",
            from_pct=0.0,
            to_pct=0.0,
            reason="promotion",
        )
        # Round-trip JSON
        s = json.dumps(r.to_dict())
        loaded = json.loads(s)
        assert loaded["kind"] == "transition"


class TestRecordDenied:
    def test_required_fields(self) -> None:
        r = record_denied(
            actor="ci",
            attempted_from="shadow",
            attempted_to="active",
            reason="promotion_gate_failed",
            failed_checks=["canary_anomalies_remaining=1 > 0", "MAE delta too high"],
        )
        assert r.kind == AuditKind.DENIED
        assert r.from_stage == "shadow"
        assert r.to_stage == "active"
        assert r.reason == "promotion_gate_failed"
        assert r.failed_checks == (
            "canary_anomalies_remaining=1 > 0",
            "MAE delta too high",
        )
        # No promotion report / gate_result on a denied record.
        assert r.gate_result is None
        assert r.promotion_report is None

    def test_with_config_hash_and_commit(self) -> None:
        r = record_denied(
            actor="ci",
            attempted_from="shadow",
            attempted_to="active",
            reason="config_drift",
            failed_checks=["mode mismatch: continuous vs bucket"],
            commit_sha="abc1234",
            config_hash="sha256:...",
        )
        d = r.to_dict()
        assert d["kind"] == "denied"
        assert d["commit_sha"] == "abc1234"
        assert d["config_hash"] == "sha256:..."
        assert d["failed_checks"] == ["mode mismatch: continuous vs bucket"]


# ── AuditLog ──────────────────────────────────────────────────────────────


class TestAuditLog:
    def test_empty_initially(self) -> None:
        log = AuditLog()
        assert len(log) == 0
        assert log.records == []

    def test_append_increments(self) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        assert len(log) == 1
        assert log.records[0].kind == AuditKind.TRANSITION

    def test_rejects_non_audit_record(self) -> None:
        log = AuditLog()
        with pytest.raises(TypeError, match="AuditRecord"):
            log.append({"kind": "transition"})  # type: ignore[arg-type]

    def test_records_returns_defensive_copy(self) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        snapshot = log.records
        snapshot.clear()
        # Internal state untouched
        assert len(log) == 1

    def test_by_kind_filters(self) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        log.append(
            record_denied(
                actor="ci",
                attempted_from="shadow",
                attempted_to="active",
                reason="canary",
                failed_checks=["a"],
            )
        )
        assert len(log.by_kind(AuditKind.TRANSITION)) == 1
        assert len(log.by_kind(AuditKind.DENIED)) == 1
        assert len(log.by_kind(AuditKind.ROLLBACK)) == 0

    def test_to_dicts_round_trip(self) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        d = log.to_dicts()
        assert d[0]["kind"] == "transition"

    def test_clear_resets(self) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        log.clear()
        assert len(log) == 0


# ── JSONL writer ──────────────────────────────────────────────────────────


class TestWriteAuditLog:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        log.append(
            record_denied(
                actor="ci",
                attempted_from="shadow",
                attempted_to="active",
                reason="canary",
                failed_checks=["a"],
            )
        )
        path = tmp_path / "audit.jsonl"
        write_audit_log(log, path)
        text = path.read_text(encoding="utf-8")
        # Two lines, both valid JSON
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 2
        for ln in lines:
            obj = json.loads(ln)
            assert "kind" in obj

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text('{"kind":"transition","timestamp":"t1"}\n', encoding="utf-8")
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
                timestamp="t2",
            )
        )
        write_audit_log(log, path)
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        # Original preserved
        assert json.loads(lines[0])["timestamp"] == "t1"
        assert json.loads(lines[1])["timestamp"] == "t2"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        path = tmp_path / "nested" / "dir" / "audit.jsonl"
        write_audit_log(log, path)
        assert path.is_file()

    def test_read_audit_log_round_trip(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
                commit_sha="abc",
            )
        )
        path = tmp_path / "audit.jsonl"
        write_audit_log(log, path)
        records = read_audit_log(path)
        assert len(records) == 1
        assert records[0]["commit_sha"] == "abc"
        assert records[0]["kind"] == "transition"

    def test_read_handles_missing_file(self, tmp_path: Path) -> None:
        assert read_audit_log(tmp_path / "nope.jsonl") == []


# ── Integration with RolloutController events ─────────────────────────────


class TestControllerIntegration:
    def test_records_from_controller_events(self) -> None:
        ctrl = RolloutController(flag=FeatureFlag.PER90_SHRINKAGE)
        ctrl.promote(new_stage=FlagStage.SHADOW)
        ctrl.promote(new_stage=FlagStage.ACTIVE, new_rollout_pct=25.0)

        records = records_from_controller_events(
            ctrl.events,
            actor="ml-training",
            commit_sha="abc1234",
        )
        assert len(records) == 2
        assert records[0].from_stage == "disabled"
        assert records[0].to_stage == "shadow"
        assert records[0].reason == "promotion"
        assert records[0].commit_sha == "abc1234"
        assert records[1].from_stage == "shadow"
        assert records[1].to_stage == "active"
        assert records[1].to_pct == 25.0

    def test_records_attach_config_hash_per_event(self) -> None:
        ctrl = RolloutController(flag=FeatureFlag.PER90_SHRINKAGE)
        ctrl.promote(
            new_stage=FlagStage.SHADOW,
            config_snapshot={"stage": "shadow", "pct": 0.0},
        )
        ctrl.promote(
            new_stage=FlagStage.ACTIVE,
            new_rollout_pct=25.0,
            config_snapshot={"stage": "active", "pct": 25.0},
        )

        # Collect config_hashes from the controller events
        config_hashes = {
            str(i): ev["config_hash"]
            for i, ev in enumerate(ctrl.events)
            if "config_hash" in ev
        }
        records = records_from_controller_events(
            ctrl.events,
            actor="ml-training",
            config_hash_for_event=config_hashes,
        )
        assert len(records) == 2
        assert records[0].config_hash is not None
        assert records[0].config_hash.startswith("sha256:")
        assert records[1].config_hash is not None
        # The two transitions have different snapshots → different hashes
        assert records[0].config_hash != records[1].config_hash

    def test_end_to_end_audit_trail(self, tmp_path: Path) -> None:
        """Build a full audit trail from controller events and persist it."""
        ctrl = RolloutController(flag=FeatureFlag.PER90_SHRINKAGE)
        ctrl.promote(
            new_stage=FlagStage.SHADOW,
            config_snapshot={"stage": "shadow", "pct": 0.0},
        )
        # Simulate a denied attempt to ACTIVE
        denied = record_denied(
            actor="ml-training",
            attempted_from="shadow",
            attempted_to="active",
            reason="promotion_gate_failed",
            failed_checks=["canary_anomalies_remaining=1 > 0"],
            commit_sha="abc1234",
            config_hash="sha256:abc",
        )
        audit = AuditLog()
        audit.extend(  # type: ignore[attr-defined]
            records_from_controller_events(
                ctrl.events,
                actor="ml-training",
                commit_sha="abc1234",
            )
        )
        audit.append(denied)
        assert len(audit) == 2
        path = tmp_path / "audit.jsonl"
        write_audit_log(audit, path)
        records = read_audit_log(path)
        assert records[0]["kind"] == "transition"
        assert records[0]["from_stage"] == "disabled"
        assert records[1]["kind"] == "denied"
        assert records[1]["failed_checks"] == ["canary_anomalies_remaining=1 > 0"]


# ── Mandatory negative cases from plan §17 ────────────────────────────────


class TestMandatoryNegative:
    def test_denied_record_carries_failed_checks(self) -> None:
        """A denied transition must record its reason and failed checks."""
        r = record_denied(
            actor="ci",
            attempted_from="shadow",
            attempted_to="active",
            reason="config_drift",
            failed_checks=[
                "mode mismatch: continuous vs bucket",
                "canary_anomalies_remaining=2 > 0",
            ],
            commit_sha="abc",
        )
        assert r.failed_checks
        assert "mode mismatch" in r.failed_checks[0]

    def test_transition_record_carries_config_hash(self) -> None:
        """Successful transitions must include the config_hash."""
        r = record_transition(
            actor="ci",
            flag="enable_shrinkage",
            from_stage="shadow",
            to_stage="active",
            from_pct=10.0,
            to_pct=25.0,
            reason="promotion",
            config_hash="sha256:cafebabe",
            commit_sha="abc",
        )
        d = r.to_dict()
        assert d["config_hash"] == "sha256:cafebabe"
        assert d["commit_sha"] == "abc"

    def test_audit_log_must_be_append_only(self) -> None:
        """The log must reject mutation of existing records."""
        log = AuditLog()
        log.append(
            record_transition(
                actor="ci",
                flag="x",
                from_stage="disabled",
                to_stage="shadow",
                from_pct=0.0,
                to_pct=0.0,
                reason="promotion",
            )
        )
        # Records returned by `records` are a defensive copy
        snapshot = log.records
        snapshot[0] = "tampered"  # type: ignore[assignment]
        # Internal state untouched
        assert isinstance(log.records[0], AuditRecord)
