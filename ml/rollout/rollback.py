"""Production-grade rollback executor (WS17 of plan.md).

Two operations are supported:

* :func:`rollback_all_to_disabled` — atomic kill-switch that forces
  every known feature flag to ``DISABLED`` with ``rollout_pct=0``.
  This is the operator's emergency button: the fastest possible
  return to the safe legacy path.

* :func:`rollback_to_snapshot` — restores a previously captured
  :class:`Snapshot` (:mod:`ml.rollout.snapshots`) flag-by-flag.
  Used when the operator wants to keep some flags ACTIVE (the ones
  that did not regress) and only rewind the ones that did.

Both operations are:

* **Idempotent** — running the same rollback twice yields the same
  end-state and appends the same kind of audit record.  A no-op
  rollback (everything already at the target stage) is still
  recorded with ``idempotent=True`` for traceability.
* **Auditable** — every flag affected by a rollback is recorded as
  an :class:`AuditKind.ROLLBACK` entry with ``from_stage``,
  ``from_pct``, ``to_stage`` (``disabled`` or the snapshot value),
  ``to_pct``, ``actor``, ``commit_sha``, and (when applicable) the
  ``snapshot_name`` and ``trigger``.
* **Fast** — there is no I/O inside the executor itself; the caller
  persists the state and audit log after the executor returns.
* **Testable** — the executor is a pure function of its inputs
  (state, audit_log, args).  Tests construct a state in memory,
  invoke the executor, and assert on the resulting state + audit.

Trigger taxonomy
----------------

``trigger`` is a free-form string identifying *why* the rollback
happened.  The plan §19 catalog is:

* ``manual`` — invoked by an operator
* ``promotion_regression`` — promotion gate would fail
* ``canary_anomaly`` — canary metrics out of band
* ``config_drift`` — runtime config drifted from snapshot
* ``runtime_error_threshold`` — observed error rate exceeded budget
* ``invariant_violation`` — critical invariant broken at runtime
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .audit import AuditLog, record_rollback
from .controller import FeatureFlag, FlagStage, RolloutController
from .snapshots import Snapshot

log = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RollbackReport:
    """Outcome of a rollback operation (WS17).

    Attributes:
        operation: ``"disable_all"`` or ``"restore_snapshot"``.
        trigger: The trigger that fired (or ``"manual"``).
        actor: Operator or system that initiated the rollback.
        commit_sha: Commit SHA at rollback time (when known).
        snapshot_name: Name of the restored snapshot (for
            ``restore_snapshot``); ``None`` for ``disable_all``.
        affected_flags: Tuple of flag names that were rolled back.
        already_at_target: Tuple of flag names that were already
            at the target state and were therefore a no-op.
        duration_ms: Wall-clock duration of the executor call in ms.
        idempotent: ``True`` when no flag actually changed stage.
            Note: this is *content idempotency*, not "called twice".
            The audit trail is always appended (one record per flag).
        started_at: ISO-8601 UTC timestamp at start.
        finished_at: ISO-8601 UTC timestamp at end.
        config_hash: SHA-256 of the restored config, when available.
        extra: Free-form metadata for the operator.
    """

    operation: str
    trigger: str
    actor: str
    affected_flags: tuple[str, ...]
    already_at_target: tuple[str, ...]
    duration_ms: float
    idempotent: bool
    started_at: str
    finished_at: str
    commit_sha: str | None = None
    snapshot_name: str | None = None
    config_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "trigger": self.trigger,
            "actor": self.actor,
            "commit_sha": self.commit_sha,
            "snapshot_name": self.snapshot_name,
            "affected_flags": list(self.affected_flags),
            "already_at_target": list(self.already_at_target),
            "duration_ms": self.duration_ms,
            "idempotent": self.idempotent,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config_hash": self.config_hash,
            "extra": dict(self.extra),
        }


# ── Validation helpers ─────────────────────────────────────────────────────


def _validate_actor(actor: str) -> str:
    if not actor or not actor.strip():
        raise ValueError("actor must be a non-empty string")
    return actor.strip()


def _validate_reason(reason: str) -> str:
    if not reason or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return reason.strip()


def _normalise_stage(stage: str) -> FlagStage:
    """Coerce a string to a known :class:`FlagStage`; raise on invalid input."""
    try:
        return FlagStage(stage)
    except ValueError as exc:
        valid = ", ".join(s.value for s in FlagStage)
        raise ValueError(
            f"unknown stage {stage!r}: valid values are {valid}"
        ) from exc


# ── Public API ─────────────────────────────────────────────────────────────


def rollback_all_to_disabled(
    *,
    state_flags: Mapping[str, Mapping[str, Any]],
    audit_log: AuditLog,
    actor: str,
    reason: str,
    commit_sha: str | None = None,
    trigger: str = "manual",
    flags_to_consider: Iterable[FeatureFlag] | None = None,
    now: str | None = None,
) -> tuple[dict[str, dict[str, Any]], RollbackReport]:
    """Force every (considered) flag to ``DISABLED`` with ``rollout_pct=0``.

    Parameters
    ----------
    state_flags:
        Current per-flag state, mapping ``flag_name → {stage, rollout_pct,
        updated_at, updated_by, note, …}``.  The mapping is *not* mutated;
        a new dict with the rolled-back values is returned.
    audit_log:
        Append-only audit log.  One :class:`AuditKind.ROLLBACK` record
        is appended for every flag that was actually rewound; a single
        record with ``extra.idempotent=True`` is appended when the
        flag was already at ``DISABLED``.
    actor:
        Operator or system that initiated the rollback.
    reason:
        Free-form human-readable reason (mandatory for traceability).
    commit_sha:
        Commit SHA at rollback time (optional).
    trigger:
        Trigger identifier (defaults to ``"manual"``).
    flags_to_consider:
        Subset of :class:`FeatureFlag` to roll back.  ``None`` means
        "every flag known to the system".  This is useful in tests
        and in partial rollbacks (e.g. only the low-sample family).
    now:
        Override for the audit timestamp; tests only.

    Returns
    -------
    (new_state, report)
        A new per-flag state dict and a :class:`RollbackReport`.
    """
    actor = _validate_actor(actor)
    reason = _validate_reason(reason)
    started = now or datetime.now(tz=timezone.utc).isoformat()
    t0 = time.perf_counter()

    considered = list(flags_to_consider) if flags_to_consider is not None else list(FeatureFlag)
    considered_values = {f.value for f in considered}

    new_state: dict[str, dict[str, Any]] = {k: dict(v) for k, v in state_flags.items()}
    affected: list[str] = []
    no_op: list[str] = []
    any_change = False

    for flag in considered:
        flag_value = flag.value
        current = new_state.get(
            flag_value,
            {"stage": FlagStage.DISABLED.value, "rollout_pct": 0.0},
        )
        from_stage = str(current.get("stage", FlagStage.DISABLED.value))
        from_pct = float(current.get("rollout_pct", 0.0))
        if from_stage == FlagStage.DISABLED.value and from_pct == 0.0:
            no_op.append(flag_value)
            audit_log.append(
                record_rollback(
                    actor=actor,
                    flag=flag_value,
                    from_stage=from_stage,
                    from_pct=from_pct,
                    to_stage=FlagStage.DISABLED.value,
                    to_pct=0.0,
                    reason=reason,
                    commit_sha=commit_sha,
                    trigger=trigger,
                    extra={"idempotent": True},
                    timestamp=now,
                )
            )
            continue
        # Build a controller matching the current state to make the
        # transition auditable through the standard event pipeline.
        controller = RolloutController(
            flag=flag,
            stage=_normalise_stage(from_stage),
            rollout_pct=from_pct,
        )
        controller.promote(
            new_stage=FlagStage.DISABLED,
            new_rollout_pct=0.0,
        )
        new_state[flag_value] = {
            **current,
            "stage": FlagStage.DISABLED.value,
            "rollout_pct": 0.0,
            "updated_at": now or datetime.now(tz=timezone.utc).isoformat(),
            "updated_by": actor,
        }
        affected.append(flag_value)
        any_change = True
        audit_log.append(
            record_rollback(
                actor=actor,
                flag=flag_value,
                from_stage=from_stage,
                from_pct=from_pct,
                to_stage=FlagStage.DISABLED.value,
                to_pct=0.0,
                reason=reason,
                commit_sha=commit_sha,
                trigger=trigger,
                timestamp=now,
            )
        )

    # Drop any state entries for flags outside the considered set
    # unless they are already DISABLED — preserve operator-authored
    # state for unrelated flags.
    for flag_name in list(new_state.keys()):
        if flag_name not in considered_values:
            entry = new_state[flag_name]
            if str(entry.get("stage", "disabled")) != FlagStage.DISABLED.value:
                log.debug(
                    "rollback_all_to_disabled: skipping %s (not in flags_to_consider)",
                    flag_name,
                )

    finished = now or datetime.now(tz=timezone.utc).isoformat()
    duration_ms = (time.perf_counter() - t0) * 1000.0
    report = RollbackReport(
        operation="disable_all",
        trigger=trigger,
        actor=actor,
        commit_sha=commit_sha,
        snapshot_name=None,
        affected_flags=tuple(affected),
        already_at_target=tuple(no_op),
        duration_ms=duration_ms,
        idempotent=not any_change,
        started_at=started,
        finished_at=finished,
        config_hash=None,
        extra={"total_considered": len(considered)},
    )
    log.info(
        "rollback_all_to_disabled: actor=%s affected=%d noop=%d duration=%.2fms",
        actor, len(affected), len(no_op), duration_ms,
    )
    return new_state, report


def rollback_to_snapshot(
    *,
    state_flags: Mapping[str, Mapping[str, Any]],
    audit_log: AuditLog,
    snapshot: Snapshot,
    actor: str,
    reason: str,
    commit_sha: str | None = None,
    trigger: str = "manual",
    flags_to_consider: Iterable[FeatureFlag] | None = None,
    now: str | None = None,
) -> tuple[dict[str, dict[str, Any]], RollbackReport]:
    """Restore per-flag state from a previously captured :class:`Snapshot`.

    Parameters mirror :func:`rollback_all_to_disabled` with two
    important differences:

    * The target stage and ``rollout_pct`` for each flag come from
      ``snapshot.flags`` (not from a fixed ``DISABLED`` value).
    * A flag not present in the snapshot is left untouched (and no
      audit record is emitted for it).  This makes
      ``rollback_to_snapshot`` safe to call against a *partial*
      snapshot that captures only the low-sample family.

    The returned state is a new dict (input is not mutated).
    """
    actor = _validate_actor(actor)
    reason = _validate_reason(reason)
    if not snapshot.flags:
        raise ValueError(
            f"snapshot '{snapshot.name}' has no flags; refusing to roll back"
        )
    started = now or datetime.now(tz=timezone.utc).isoformat()
    t0 = time.perf_counter()

    considered = list(flags_to_consider) if flags_to_consider is not None else list(FeatureFlag)
    considered_values = {f.value for f in considered}

    new_state: dict[str, dict[str, Any]] = {k: dict(v) for k, v in state_flags.items()}
    affected: list[str] = []
    no_op: list[str] = []
    any_change = False

    for flag in considered:
        flag_value = flag.value
        target = snapshot.flags.get(flag_value)
        if target is None:
            # Snapshot does not mention this flag → leave untouched.
            continue
        target_stage = _normalise_stage(str(target["stage"]))
        target_pct = float(target.get("rollout_pct", 0.0))
        if not 0.0 <= target_pct <= 100.0:
            raise ValueError(
                f"snapshot '{snapshot.name}' flag '{flag_value}' has invalid "
                f"rollout_pct={target_pct}; must be in [0, 100]"
            )
        current = new_state.get(
            flag_value,
            {"stage": FlagStage.DISABLED.value, "rollout_pct": 0.0},
        )
        from_stage = str(current.get("stage", FlagStage.DISABLED.value))
        from_pct = float(current.get("rollout_pct", 0.0))
        if from_stage == target_stage.value and from_pct == target_pct:
            no_op.append(flag_value)
            audit_log.append(
                record_rollback(
                    actor=actor,
                    flag=flag_value,
                    from_stage=from_stage,
                    from_pct=from_pct,
                    to_stage=target_stage.value,
                    to_pct=target_pct,
                    reason=reason,
                    commit_sha=commit_sha,
                    snapshot_name=snapshot.name,
                    trigger=trigger,
                    extra={"idempotent": True},
                    timestamp=now,
                )
            )
            continue
        # Apply the transition through a controller so the standard
        # invariants (DISABLED ↔ SHADOW ↔ ACTIVE, ACTIVE requires
        # rollout_pct > 0) are enforced.
        controller = RolloutController(
            flag=flag,
            stage=_normalise_stage(from_stage),
            rollout_pct=from_pct,
        )
        controller.promote(
            new_stage=target_stage,
            new_rollout_pct=target_pct,
        )
        new_state[flag_value] = {
            **current,
            "stage": target_stage.value,
            "rollout_pct": target_pct,
            "updated_at": now or datetime.now(tz=timezone.utc).isoformat(),
            "updated_by": actor,
        }
        affected.append(flag_value)
        any_change = True
        audit_log.append(
            record_rollback(
                actor=actor,
                flag=flag_value,
                from_stage=from_stage,
                from_pct=from_pct,
                to_stage=target_stage.value,
                to_pct=target_pct,
                reason=reason,
                commit_sha=commit_sha,
                snapshot_name=snapshot.name,
                trigger=trigger,
                timestamp=now,
            )
        )

    finished = now or datetime.now(tz=timezone.utc).isoformat()
    duration_ms = (time.perf_counter() - t0) * 1000.0
    report = RollbackReport(
        operation="restore_snapshot",
        trigger=trigger,
        actor=actor,
        commit_sha=commit_sha,
        snapshot_name=snapshot.name,
        affected_flags=tuple(affected),
        already_at_target=tuple(no_op),
        duration_ms=duration_ms,
        idempotent=not any_change,
        started_at=started,
        finished_at=finished,
        config_hash=snapshot.config_hash,
        extra={"total_considered": len(considered)},
    )
    log.info(
        "rollback_to_snapshot: snapshot=%s actor=%s affected=%d noop=%d duration=%.2fms",
        snapshot.name, actor, len(affected), len(no_op), duration_ms,
    )
    return new_state, report


__all__ = [
    "RollbackReport",
    "rollback_all_to_disabled",
    "rollback_to_snapshot",
]
