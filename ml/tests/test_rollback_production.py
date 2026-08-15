"""Production-grade rollback tests (WS17 of plan.md).

The test surface covers the four hard requirements from plan §19:

* **idempotent** — repeated invocations do not change the end state
  beyond the first one, and produce a stable audit trail.
* **auditable** — every affected flag is recorded as an
  :class:`AuditKind.ROLLBACK` entry with the full chain of custody.
* **fast** — a full kill-switch over every known flag completes in
  well under one second on commodity hardware.
* **testable** — the executor is a pure function of its inputs; the
  tests construct state in-memory and assert on the result.

The CLI surface is exercised via ``subprocess`` so the operator-facing
contract (exit codes, JSON shape) is locked down as well.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Importante: alcuni moduli (es. ``ml.config``) istanziano ``MLConfig`` a
# import-time.  Settiamo l'env anche qui per coerenza con gli altri test
# del modulo (vedi ``test_rollback_and_adr_guard.py``).
os.environ.setdefault("ML_DATABASE_URL", "postgresql://x:x@localhost/x")

from ml.rollout.audit import AuditKind, AuditLog  # noqa: E402
from ml.rollout.controller import FeatureFlag, FlagStage  # noqa: E402
from ml.rollout.rollback import (  # noqa: E402
    RollbackReport,
    rollback_all_to_disabled,
    rollback_to_snapshot,
)
from ml.rollout.snapshots import (  # noqa: E402
    Snapshot,
    SnapshotError,
    delete_snapshot,
    latest_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ──────────────────────────────────────────────────────────────


def _all_flag_values() -> list[str]:
    return [f.value for f in FeatureFlag]


def _mixed_state() -> dict[str, dict[str, object]]:
    """Stato misto: 2 ACTIVE, 1 SHADOW, resto DISABLED."""
    return {
        FeatureFlag.LIMITED_SAMPLE_TRAINING.value: {
            "stage": FlagStage.ACTIVE.value,
            "rollout_pct": 25.0,
        },
        FeatureFlag.PER90_SHRINKAGE.value: {
            "stage": FlagStage.ACTIVE.value,
            "rollout_pct": 100.0,
        },
        FeatureFlag.BREAKOUT_MODEL.value: {
            "stage": FlagStage.SHADOW.value,
            "rollout_pct": 0.0,
        },
        FeatureFlag.RECENT_ROLE_FEATURES.value: {
            "stage": FlagStage.DISABLED.value,
            "rollout_pct": 0.0,
        },
        FeatureFlag.RELIABILITY_WEIGHT_CONTINUOUS.value: {
            "stage": FlagStage.DISABLED.value,
            "rollout_pct": 0.0,
        },
    }


# ── audit.record_rollback() helper ────────────────────────────────────────


class TestRecordRollbackHelper:
    def test_kind_is_rollback(self) -> None:
        rec = _make_record()
        assert rec.kind == AuditKind.ROLLBACK

    def test_required_fields(self) -> None:
        rec = _make_record()
        assert rec.flag == "enable_shrinkage"
        assert rec.from_stage == "active"
        assert rec.from_pct == 100.0
        assert rec.to_stage == "disabled"
        assert rec.to_pct == 0.0
        assert rec.reason == "canary anomaly"
        assert rec.actor == "lbrunori"
        assert rec.commit_sha == "abc1234"
        assert rec.config_hash == "sha256:abc"

    def test_trigger_and_snapshot_in_extra(self) -> None:
        rec = _make_record()
        assert rec.extra.get("trigger") == "canary_anomaly"
        assert rec.extra.get("snapshot_name") is None

    def test_snapshot_name_propagates(self) -> None:
        from ml.rollout.audit import record_rollback

        rec = record_rollback(
            actor="lbrunori",
            flag="enable_shrinkage",
            from_stage="active",
            from_pct=100.0,
            to_stage="disabled",
            to_pct=0.0,
            reason="restore",
            snapshot_name="pre-shrinkage-2026-08-12",
            trigger="canary_anomaly",
        )
        assert rec.extra["snapshot_name"] == "pre-shrinkage-2026-08-12"

    def test_to_dict_serialises(self) -> None:
        rec = _make_record()
        d = rec.to_dict()
        assert d["kind"] == "rollback"
        assert d["flag"] == "enable_shrinkage"
        assert d["from_stage"] == "active"
        assert d["to_stage"] == "disabled"
        # JSON-serialisable
        json.dumps(d)


def _make_record():
    from ml.rollout.audit import record_rollback

    return record_rollback(
        actor="lbrunori",
        flag="enable_shrinkage",
        from_stage="active",
        from_pct=100.0,
        to_stage="disabled",
        to_pct=0.0,
        reason="canary anomaly",
        commit_sha="abc1234",
        config_hash="sha256:abc",
        trigger="canary_anomaly",
    )


# ── rollback_all_to_disabled ──────────────────────────────────────────────


class TestRollbackAllToDisabled:
    def test_disables_every_active_flag(self) -> None:
        audit = AuditLog()
        new_state, report = rollback_all_to_disabled(
            state_flags=_mixed_state(),
            audit_log=audit,
            actor="lbrunori",
            reason="incident TICKET-123",
            commit_sha="deadbeef",
            trigger="canary_anomaly",
        )
        for name in _all_flag_values():
            assert new_state[name]["stage"] == "disabled"
            assert new_state[name]["rollout_pct"] == 0.0
        # I 3 flag "up" sono stati effettivamente rewound.
        assert set(report.affected_flags) == {
            FeatureFlag.LIMITED_SAMPLE_TRAINING.value,
            FeatureFlag.PER90_SHRINKAGE.value,
            FeatureFlag.BREAKOUT_MODEL.value,
        }
        # I 2 flag già a DISABLED sono registrati come no-op idempotenti.
        assert set(report.already_at_target) == {
            FeatureFlag.RECENT_ROLE_FEATURES.value,
            FeatureFlag.RELIABILITY_WEIGHT_CONTINUOUS.value,
        }
        assert report.idempotent is False
        assert report.operation == "disable_all"
        assert report.trigger == "canary_anomaly"
        assert report.commit_sha == "deadbeef"

    def test_appends_audit_record_per_flag(self) -> None:
        audit = AuditLog()
        rollback_all_to_disabled(
            state_flags=_mixed_state(),
            audit_log=audit,
            actor="lbrunori",
            reason="incident",
        )
        rollback_records = audit.by_kind(AuditKind.ROLLBACK)
        # 5 record: 3 rewound + 2 no-op idempotenti.
        assert len(rollback_records) == len(_all_flag_values())
        for rec in rollback_records:
            assert rec.kind == AuditKind.ROLLBACK
            assert rec.to_stage == "disabled"
            assert rec.to_pct == 0.0
            assert rec.actor == "lbrunori"
            assert rec.reason == "incident"

    def test_idempotent_when_already_disabled(self) -> None:
        """Tutti DISABLED → secondo rollback non cambia nulla (no-op)."""
        already: dict[str, dict[str, object]] = {
            f.value: {"stage": "disabled", "rollout_pct": 0.0}
            for f in FeatureFlag
        }
        audit1 = AuditLog()
        _, r1 = rollback_all_to_disabled(
            state_flags=already,
            audit_log=audit1,
            actor="lbrunori",
            reason="first",
        )
        assert r1.idempotent is True
        assert set(r1.already_at_target) == set(already)
        assert r1.affected_flags == ()

        audit2 = AuditLog()
        _, r2 = rollback_all_to_disabled(
            state_flags=already,
            audit_log=audit2,
            actor="lbrunori",
            reason="second",
        )
        assert r2.idempotent is True
        assert r2.affected_flags == ()
        # Ogni esecuzione registra un record per flag (audit trail integro).
        assert len(audit2.by_kind(AuditKind.ROLLBACK)) == len(_all_flag_values())

    def test_running_twice_converges_to_same_state(self) -> None:
        """Doppia esecuzione → stesso stato finale."""
        state = _mixed_state()
        a1 = AuditLog()
        s1, _ = rollback_all_to_disabled(
            state_flags=state, audit_log=a1, actor="o", reason="r1"
        )
        a2 = AuditLog()
        s2, _ = rollback_all_to_disabled(
            state_flags=s1, audit_log=a2, actor="o", reason="r2"
        )
        assert s1 == s2

    def test_fast_under_one_second(self) -> None:
        """Anche con tutti i flag attivi, il kill-switch deve restare <1s."""
        state = {
            f.value: {"stage": "active", "rollout_pct": 50.0}
            for f in FeatureFlag
        }
        audit = AuditLog()
        t0 = time.perf_counter()
        _, report = rollback_all_to_disabled(
            state_flags=state,
            audit_log=audit,
            actor="o",
            reason="speed check",
        )
        elapsed = time.perf_counter() - t0
        # assertion esplicita sul campo duration_ms
        assert report.duration_ms < 1000.0, (
            f"rollback took {report.duration_ms:.1f}ms (must be < 1000ms)"
        )
        assert elapsed < 1.0

    def test_rejects_empty_actor(self) -> None:
        with pytest.raises(ValueError, match="actor"):
            rollback_all_to_disabled(
                state_flags={},
                audit_log=AuditLog(),
                actor="   ",
                reason="r",
            )

    def test_rejects_empty_reason(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            rollback_all_to_disabled(
                state_flags={},
                audit_log=AuditLog(),
                actor="o",
                reason="",
            )

    def test_does_not_mutate_input_state(self) -> None:
        state = _mixed_state()
        snapshot = json.dumps(state, sort_keys=True, default=str)
        rollback_all_to_disabled(
            state_flags=state,
            audit_log=AuditLog(),
            actor="o",
            reason="r",
        )
        assert json.dumps(state, sort_keys=True, default=str) == snapshot

    def test_partial_rollback(self) -> None:
        """``flags_to_consider`` limita il rollback a una sotto-famiglia."""
        audit = AuditLog()
        new_state, report = rollback_all_to_disabled(
            state_flags=_mixed_state(),
            audit_log=audit,
            actor="o",
            reason="r",
            flags_to_consider=[FeatureFlag.PER90_SHRINKAGE],
        )
        # Solo il flag considerato viene rewound.
        assert report.affected_flags == (FeatureFlag.PER90_SHRINKAGE.value,)
        # Gli altri flag mantengono lo stage originale.
        assert (
            new_state[FeatureFlag.LIMITED_SAMPLE_TRAINING.value]["stage"]
            == "active"
        )


# ── rollback_to_snapshot ──────────────────────────────────────────────────


class TestRollbackToSnapshot:
    def _snapshot(self) -> Snapshot:
        return Snapshot(
            name="pre-shrinkage",
            saved_at="2026-08-12T10:00:00+00:00",
            saved_by="lbrunori",
            commit_sha="abc1234",
            config_hash="sha256:abc",
            flags={
                FeatureFlag.PER90_SHRINKAGE.value: {
                    "stage": "disabled",
                    "rollout_pct": 0.0,
                },
                FeatureFlag.BREAKOUT_MODEL.value: {
                    "stage": "active",
                    "rollout_pct": 25.0,
                },
            },
        )

    def test_restores_snapshot_state(self) -> None:
        snap = self._snapshot()
        state = _mixed_state()
        audit = AuditLog()
        new_state, report = rollback_to_snapshot(
            state_flags=state,
            audit_log=audit,
            snapshot=snap,
            actor="lbrunori",
            reason="restore",
            commit_sha="deadbeef",
            trigger="promotion_regression",
        )
        # I flag nello snapshot sono ripristinati.
        assert (
            new_state[FeatureFlag.PER90_SHRINKAGE.value]["stage"] == "disabled"
        )
        assert (
            new_state[FeatureFlag.BREAKOUT_MODEL.value]["stage"] == "active"
        )
        assert new_state[FeatureFlag.BREAKOUT_MODEL.value]["rollout_pct"] == 25.0
        assert report.snapshot_name == "pre-shrinkage"
        assert report.operation == "restore_snapshot"
        assert report.config_hash == "sha256:abc"

    def test_preserves_flags_not_in_snapshot(self) -> None:
        """I flag non menzionati nello snapshot restano inalterati."""
        snap = self._snapshot()  # contiene solo 2 flag
        state = _mixed_state()
        original_limited = dict(state[FeatureFlag.LIMITED_SAMPLE_TRAINING.value])
        audit = AuditLog()
        new_state, _ = rollback_to_snapshot(
            state_flags=state,
            audit_log=audit,
            snapshot=snap,
            actor="o",
            reason="r",
        )
        # LIMITED_SAMPLE_TRAINING non è nello snapshot → preservato.
        assert (
            new_state[FeatureFlag.LIMITED_SAMPLE_TRAINING.value]["stage"]
            == original_limited["stage"]
        )
        assert (
            new_state[FeatureFlag.LIMITED_SAMPLE_TRAINING.value]["rollout_pct"]
            == original_limited["rollout_pct"]
        )

    def test_audit_records_carry_snapshot_name(self) -> None:
        snap = self._snapshot()
        audit = AuditLog()
        rollback_to_snapshot(
            state_flags=_mixed_state(),
            audit_log=audit,
            snapshot=snap,
            actor="o",
            reason="r",
        )
        records = audit.by_kind(AuditKind.ROLLBACK)
        assert records, "expected at least one rollback record"
        for rec in records:
            assert rec.extra.get("snapshot_name") == "pre-shrinkage"

    def test_idempotent_when_already_at_snapshot(self) -> None:
        snap = self._snapshot()
        state = {
            FeatureFlag.PER90_SHRINKAGE.value: {
                "stage": "disabled",
                "rollout_pct": 0.0,
            },
            FeatureFlag.BREAKOUT_MODEL.value: {
                "stage": "active",
                "rollout_pct": 25.0,
            },
        }
        audit = AuditLog()
        _, report = rollback_to_snapshot(
            state_flags=state,
            audit_log=audit,
            snapshot=snap,
            actor="o",
            reason="r",
        )
        assert report.idempotent is True
        assert report.affected_flags == ()
        # I no-op sono comunque registrati per audit trail.
        assert len(audit.by_kind(AuditKind.ROLLBACK)) == 2

    def test_rejects_empty_snapshot(self) -> None:
        empty = Snapshot(
            name="empty",
            saved_at="2026-08-12T10:00:00+00:00",
            saved_by="o",
            flags={},
        )
        with pytest.raises(ValueError, match="no flags"):
            rollback_to_snapshot(
                state_flags={},
                audit_log=AuditLog(),
                snapshot=empty,
                actor="o",
                reason="r",
            )

    def test_rejects_invalid_rollout_pct(self) -> None:
        bad = Snapshot(
            name="bad",
            saved_at="2026-08-12T10:00:00+00:00",
            saved_by="o",
            flags={
                FeatureFlag.PER90_SHRINKAGE.value: {
                    "stage": "active",
                    "rollout_pct": 150.0,  # invalid
                },
            },
        )
        with pytest.raises(ValueError, match="rollout_pct"):
            rollback_to_snapshot(
                state_flags={},
                audit_log=AuditLog(),
                snapshot=bad,
                actor="o",
                reason="r",
            )

    def test_rejects_invalid_stage(self) -> None:
        bad = Snapshot(
            name="bad",
            saved_at="2026-08-12T10:00:00+00:00",
            saved_by="o",
            flags={
                FeatureFlag.PER90_SHRINKAGE.value: {
                    "stage": "banana",
                    "rollout_pct": 0.0,
                },
            },
        )
        with pytest.raises(ValueError, match="unknown stage"):
            rollback_to_snapshot(
                state_flags={},
                audit_log=AuditLog(),
                snapshot=bad,
                actor="o",
                reason="r",
            )


# ── Snapshot I/O ──────────────────────────────────────────────────────────


class TestSnapshotsIO:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        flags = {
            "enable_shrinkage": {"stage": "active", "rollout_pct": 50.0},
            "enable_breakout_model": {"stage": "shadow", "rollout_pct": 0.0},
        }
        snap = save_snapshot(
            artifacts_root=tmp_path,
            name="pre-deploy-001",
            flags=flags,
            saved_by="lbrunori",
            commit_sha="abc",
        )
        loaded = load_snapshot(tmp_path, "pre-deploy-001")
        assert loaded.name == snap.name
        assert loaded.saved_by == "lbrunori"
        assert loaded.commit_sha == "abc"
        assert loaded.flags["enable_shrinkage"]["stage"] == "active"
        assert loaded.flags["enable_shrinkage"]["rollout_pct"] == 50.0
        assert loaded.flags["enable_breakout_model"]["stage"] == "shadow"

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        flags_v1 = {
            "enable_shrinkage": {"stage": "active", "rollout_pct": 10.0},
        }
        flags_v2 = {
            "enable_shrinkage": {"stage": "active", "rollout_pct": 25.0},
        }
        save_snapshot(
            artifacts_root=tmp_path,
            name="pre",
            flags=flags_v1,
            saved_by="o",
        )
        save_snapshot(
            artifacts_root=tmp_path,
            name="pre",
            flags=flags_v2,
            saved_by="o",
        )
        loaded = load_snapshot(tmp_path, "pre")
        assert loaded.flags["enable_shrinkage"]["rollout_pct"] == 25.0

    def test_list_snapshots_sorted_desc(self, tmp_path: Path) -> None:
        # Salvataggio in ordine deterministico (timestamps realistici).
        for ts, pct in [("2026-08-10T00:00:00+00:00", 10.0), ("2026-08-12T00:00:00+00:00", 25.0), ("2026-08-11T00:00:00+00:00", 15.0)]:
            save_snapshot(
                artifacts_root=tmp_path,
                name=f"snap-{ts[:10]}",
                flags={"f": {"stage": "active", "rollout_pct": pct}},
                saved_by="o",
                saved_at=ts,
            )
        snaps = list_snapshots(tmp_path)
        assert [s.saved_at for s in snaps] == [
            "2026-08-12T00:00:00+00:00",
            "2026-08-11T00:00:00+00:00",
            "2026-08-10T00:00:00+00:00",
        ]
        latest = latest_snapshot(tmp_path)
        assert latest is not None
        assert latest.saved_at == "2026-08-12T00:00:00+00:00"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="not found"):
            load_snapshot(tmp_path, "ghost")

    def test_list_skips_malformed_files(self, tmp_path: Path) -> None:
        # Crea un file malformato accanto a uno valido.
        save_snapshot(
            artifacts_root=tmp_path,
            name="good",
            flags={"f": {"stage": "disabled", "rollout_pct": 0.0}},
            saved_by="o",
        )
        (tmp_path / "snapshots" / "bad.json").write_text("{not valid json", encoding="utf-8")
        snaps = list_snapshots(tmp_path)
        # Il valido è incluso, il malformato è skippato con warning.
        assert [s.name for s in snaps] == ["good"]

    def test_delete_is_idempotent(self, tmp_path: Path) -> None:
        save_snapshot(
            artifacts_root=tmp_path,
            name="x",
            flags={"f": {"stage": "disabled", "rollout_pct": 0.0}},
            saved_by="o",
        )
        assert delete_snapshot(tmp_path, "x") is True
        # Seconda cancellazione → False, no raise.
        assert delete_snapshot(tmp_path, "x") is False

    def test_validation_rejects_bad_name(self, tmp_path: Path) -> None:
        for bad in ["", "../escape", "with space", "a" * 200]:
            with pytest.raises(SnapshotError):
                save_snapshot(
                    artifacts_root=tmp_path,
                    name=bad,
                    flags={"f": {"stage": "disabled", "rollout_pct": 0.0}},
                    saved_by="o",
                )

    def test_validation_rejects_empty_actor(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="saved_by"):
            save_snapshot(
                artifacts_root=tmp_path,
                name="x",
                flags={"f": {"stage": "disabled", "rollout_pct": 0.0}},
                saved_by="   ",
            )

    def test_validation_rejects_flag_without_stage(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="stage"):
            save_snapshot(
                artifacts_root=tmp_path,
                name="x",
                flags={"f": {"rollout_pct": 0.0}},  # missing 'stage'
                saved_by="o",
            )

    def test_save_computes_config_hash(self, tmp_path: Path) -> None:
        snap = save_snapshot(
            artifacts_root=tmp_path,
            name="x",
            flags={"f": {"stage": "disabled", "rollout_pct": 0.0}},
            saved_by="o",
            config_data={"enable_shrinkage": True},
        )
        assert snap.config_hash is not None
        assert snap.config_hash.startswith("sha256:")


# ── CLI: run_rollout rollback-all / save-snapshot / restore-snapshot ────


def _run_cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Invoca ``python -m ml.run_rollout`` con ``args`` e ritorna il completed process."""
    cmd = [sys.executable, "-m", "ml.run_rollout", *args]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=60,
    )


class TestRunRolloutCLI:
    def test_rollback_all_disables_every_flag(self, tmp_path: Path) -> None:
        # Pre-popola lo stato con un flag ACTIVE.
        state = {
            "version": 1,
            "updated_at": "2026-08-12T00:00:00+00:00",
            "flags": {
                "enable_shrinkage": {
                    "flag": "enable_shrinkage",
                    "stage": "active",
                    "rollout_pct": 100.0,
                    "updated_at": "2026-08-12T00:00:00+00:00",
                    "updated_by": "ci",
                    "note": "",
                },
                "enable_breakout_model": {
                    "flag": "enable_breakout_model",
                    "stage": "shadow",
                    "rollout_pct": 0.0,
                    "updated_at": "2026-08-12T00:00:00+00:00",
                    "updated_by": "ci",
                    "note": "",
                },
            },
            "audit": [],
        }
        (tmp_path / "rollout").mkdir()
        (tmp_path / "rollout" / "state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "rollback-all",
            "--reason", "canary_anomaly",
            "--actor", "lbrunori",
            "--trigger", "canary_anomaly",
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["status"] == "ok"
        report = payload["report"]
        assert report["operation"] == "disable_all"
        assert "enable_shrinkage" in report["affected_flags"]
        # Stato persistito: tutti DISABLED.
        new_state = json.loads(
            (tmp_path / "rollout" / "state.json").read_text(encoding="utf-8")
        )
        for flag in new_state["flags"].values():
            assert flag["stage"] == "disabled"
            assert flag["rollout_pct"] == 0.0
        # Audit trail contiene record ROLLBACK.
        assert any(
            ev.get("kind") == "rollback" for ev in new_state["audit"]
        )

    def test_rollback_all_requires_reason(self, tmp_path: Path) -> None:
        # Senza ``--reason`` argparse restituisce exit 2 con un usage error.
        # Verifichiamo che lo stderr segnali l'argomento mancante.
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "rollback-all",
            "--actor", "lbrunori",
        )
        assert proc.returncode == 2
        assert "reason" in proc.stderr

    def test_save_snapshot_then_restore(self, tmp_path: Path) -> None:
        # Stato iniziale con un flag DISABLED (cmd_save_snapshot rifiuta
        # di snapshotare uno stato completamente vuoto).
        (tmp_path / "rollout").mkdir()
        (tmp_path / "rollout" / "state.json").write_text(
            json.dumps({
                "version": 1,
                "updated_at": "2026-08-12T00:00:00+00:00",
                "flags": {
                    "enable_shrinkage": {
                        "flag": "enable_shrinkage",
                        "stage": "disabled",
                        "rollout_pct": 0.0,
                        "updated_at": "2026-08-12T00:00:00+00:00",
                        "updated_by": "ci",
                        "note": "",
                    },
                },
                "audit": [],
            }),
            encoding="utf-8",
        )

        # 1) Salva snapshot (stato DISABLED).
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "save-snapshot",
            "--name", "pre-test",
            "--actor", "lbrunori",
        )
        assert proc.returncode == 0, proc.stderr
        snap_payload = json.loads(proc.stdout)
        assert snap_payload["status"] == "ok"
        assert snap_payload["snapshot"]["name"] == "pre-test"

        # 2) Forza uno stato attivo manualmente (scrittura diretta).
        state_path = tmp_path / "rollout" / "state.json"
        st = json.loads(state_path.read_text(encoding="utf-8"))
        st["flags"] = {
            "enable_shrinkage": {
                "flag": "enable_shrinkage",
                "stage": "active",
                "rollout_pct": 100.0,
                "updated_at": "2026-08-12T00:00:00+00:00",
                "updated_by": "ci",
                "note": "",
            },
        }
        state_path.write_text(json.dumps(st), encoding="utf-8")

        # 3) Ripristina dallo snapshot (deve tornare a DISABLED).
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "restore-snapshot",
            "--name", "pre-test",
            "--reason", "test",
            "--actor", "lbrunori",
        )
        assert proc.returncode == 0, proc.stderr
        st_after = json.loads(state_path.read_text(encoding="utf-8"))
        assert st_after["flags"]["enable_shrinkage"]["stage"] == "disabled"
        assert st_after["flags"]["enable_shrinkage"]["rollout_pct"] == 0.0
        # Audit trail contiene il rollback con snapshot_name.
        rollbacks = [
            ev for ev in st_after["audit"]
            if ev.get("kind") == "rollback"
        ]
        assert rollbacks, "expected at least one rollback audit event"
        assert any(
            ev.get("extra", {}).get("snapshot_name") == "pre-test"
            for ev in rollbacks
        )

    def test_list_snapshots_empty(self, tmp_path: Path) -> None:
        (tmp_path / "rollout").mkdir()
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "list-snapshots",
        )
        assert proc.returncode == 0
        payload = json.loads(proc.stdout)
        assert payload["total"] == 0
        assert payload["snapshots"] == []

    def test_restore_snapshot_missing_name(self, tmp_path: Path) -> None:
        (tmp_path / "rollout").mkdir()
        proc = _run_cli(
            "--artifacts-dir", str(tmp_path),
            "restore-snapshot",
            "--name", "ghost",
            "--reason", "r",
            "--actor", "o",
        )
        assert proc.returncode == 2
        payload = json.loads(proc.stderr.strip().splitlines()[-1])
        assert payload["error"] == "snapshot_error"


# ── CLI: ml.scripts.rollback wrapper ──────────────────────────────────────


class TestScriptsRollbackWrapper:
    def test_disable_propagates_args(self, tmp_path: Path) -> None:
        (tmp_path / "rollout").mkdir()
        (tmp_path / "rollout" / "state.json").write_text(
            json.dumps({"version": 1, "updated_at": "", "flags": {}, "audit": []}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-m", "ml.scripts.rollback", "disable",
             "--artifacts-dir", str(tmp_path),
             "--reason", "wrapper test",
             "--actor", "lbrunori"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["status"] == "ok"
        assert payload["report"]["operation"] == "disable_all"
