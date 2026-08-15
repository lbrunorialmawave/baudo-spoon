"""Append-only audit trail for rollout transitions (WS15 of plan.md).

Every stage change — successful or denied — is persisted as a JSONL
record.  The trail is the canonical answer to *"who promoted what,
when, with which evidence, and what would have happened if we had
tried?"*.

Record format (plan §17)
------------------------
Successful transition::

    {
      "kind": "transition",
      "timestamp": "2026-08-15T11:00:00+00:00",
      "actor": "ci-bot",
      "flag": "enable_shrinkage",
      "from_stage": "shadow",
      "to_stage": "active",
      "from_pct": 10.0,
      "to_pct": 25.0,
      "reason": "promotion",
      "commit_sha": "abc1234",
      "promotion_report": "ml/reports/promotion_report.json",
      "gate_result": "PASS",
      "config_hash": "sha256:..."
    }

Denied transition::

    {
      "kind": "denied",
      "timestamp": "...",
      "actor": "ci-bot",
      "attempted_transition": {"from": "shadow", "to": "active"},
      "reason": "promotion_gate_failed",
      "failed_checks": ["canary_anomalies_remaining=1 > 0", "..."],
      "commit_sha": "...",
      "config_hash": "..."
    }

The module is I/O-free by default: callers compose the records and
decide when / how to persist them.  A small JSONL writer
(:func:`write_audit_log`) is provided for the CI workflow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger(__name__)


# ── Record taxonomy ────────────────────────────────────────────────────────


class AuditKind(str, Enum):
    """Closed set of audit record categories."""

    TRANSITION = "transition"   # successful stage change
    DENIED = "denied"           # attempted but blocked
    ROLLBACK = "rollback"       # explicit rewind (emergency / scheduled)


# ── Record types ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One row of the audit trail.

    Frozen + slotted so accidental mutation breaks the chain-of-custody.
    Use the :class:`AuditLog` builder rather than constructing directly
    for production writes.
    """

    kind: AuditKind
    timestamp: str
    actor: str
    flag: str | None
    from_stage: str | None
    to_stage: str | None
    from_pct: float | None
    to_pct: float | None
    reason: str
    commit_sha: str | None = None
    promotion_report: str | None = None
    gate_result: str | None = None
    config_hash: str | None = None
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Enums become str via asdict, but we double-coerce for clarity.
        d["kind"] = self.kind.value
        # JSON serialisation expects lists, not tuples.  Normalise here so
        # callers don't have to remember.
        if isinstance(d.get("failed_checks"), tuple):
            d["failed_checks"] = list(d["failed_checks"])
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def record_transition(
    *,
    actor: str,
    flag: str,
    from_stage: str,
    to_stage: str,
    from_pct: float,
    to_pct: float,
    reason: str,
    commit_sha: str | None = None,
    promotion_report: str | None = None,
    gate_result: str | None = None,
    config_hash: str | None = None,
    extra: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> AuditRecord:
    """Build a :class:`AuditRecord` for a *successful* transition."""
    return AuditRecord(
        kind=AuditKind.TRANSITION,
        timestamp=timestamp or _utc_now_iso(),
        actor=actor,
        flag=flag,
        from_stage=from_stage,
        to_stage=to_stage,
        from_pct=from_pct,
        to_pct=to_pct,
        reason=reason,
        commit_sha=commit_sha,
        promotion_report=promotion_report,
        gate_result=gate_result,
        config_hash=config_hash,
        extra=dict(extra or {}),
    )


def record_denied(
    *,
    actor: str,
    attempted_from: str,
    attempted_to: str,
    reason: str,
    failed_checks: Iterable[str],
    commit_sha: str | None = None,
    config_hash: str | None = None,
    flag: str | None = None,
    extra: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> AuditRecord:
    """Build a :class:`AuditRecord` for a *denied* transition."""
    return AuditRecord(
        kind=AuditKind.DENIED,
        timestamp=timestamp or _utc_now_iso(),
        actor=actor,
        flag=flag,
        from_stage=attempted_from,
        to_stage=attempted_to,
        from_pct=None,
        to_pct=None,
        reason=reason,
        commit_sha=commit_sha,
        config_hash=config_hash,
        failed_checks=tuple(failed_checks),
        extra=dict(extra or {}),
    )


# ── Append-only log ───────────────────────────────────────────────────────


@dataclass
class AuditLog:
    """In-memory append-only log of audit records.

    The list is intentionally exposed read-only via :meth:`records` and
    the JSONL writer (:func:`write_audit_log`) for production.  Tests
    inspect the in-memory list directly.
    """

    _records: list[AuditRecord] = field(default_factory=list)

    def append(self, record: AuditRecord) -> None:
        if not isinstance(record, AuditRecord):
            raise TypeError(f"AuditLog accepts AuditRecord, got {type(record).__name__}")
        self._records.append(record)
        log.info(
            "audit: %s flag=%s %s→%s actor=%s",
            record.kind.value,
            record.flag,
            record.from_stage,
            record.to_stage,
            record.actor,
        )

    def extend(self, records: Iterable[AuditRecord]) -> None:
        """Append many records in order; each is validated."""
        for r in records:
            self.append(r)

    @property
    def records(self) -> list[AuditRecord]:
        """Return a defensive copy of the records."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def by_kind(self, kind: AuditKind) -> list[AuditRecord]:
        return [r for r in self._records if r.kind == kind]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def clear(self) -> None:
        """Reset the log.  Production code should never call this."""
        self._records.clear()


# ── JSONL writer ───────────────────────────────────────────────────────────


def write_audit_log(log_obj: AuditLog, path: Path | str) -> Path:
    """Persist the audit log to disk in line-delimited JSON.

    The file is opened in append mode if it already exists so a CI
    workflow can stream records across runs without losing history.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in log_obj.records:
            fh.write(record.to_json())
            fh.write("\n")
    log.info("audit log written: %d records → %s", len(log_obj), path)
    return path


def read_audit_log(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL audit log back into a list of dicts (best-effort)."""
    path = Path(path)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("audit log: skipping malformed line: %s", exc)
    return records


# ── Integration with RolloutController events ─────────────────────────────


def records_from_controller_events(
    events: Iterable[Mapping[str, Any]],
    *,
    actor: str,
    commit_sha: str | None = None,
    config_hash_for_event: Mapping[str, str] | None = None,
) -> list[AuditRecord]:
    """Convert :class:`RolloutController` events into audit records.

    ``config_hash_for_event`` is an optional mapping ``event_index →
    config_hash`` (e.g. produced by re-hashing snapshots supplied to
    :meth:`RolloutController.promote`).  When present, the hash is
    attached to the corresponding audit record.
    """
    records: list[AuditRecord] = []
    for idx, ev in enumerate(events):
        ch = (config_hash_for_event or {}).get(str(idx))
        record = record_transition(
            actor=actor,
            flag=str(ev.get("flag", "?")),
            from_stage=str(ev.get("from_stage", "?")),
            to_stage=str(ev.get("to_stage", "?")),
            from_pct=float(ev.get("from_pct", 0.0)),
            to_pct=float(ev.get("to_pct", 0.0)),
            reason=str(ev.get("reason", "?")),
            commit_sha=commit_sha,
            config_hash=ch,
        )
        records.append(record)
    return records


# ── Helpers ────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


__all__ = [
    "AuditKind",
    "AuditRecord",
    "AuditLog",
    "record_transition",
    "record_denied",
    "write_audit_log",
    "read_audit_log",
    "records_from_controller_events",
]
