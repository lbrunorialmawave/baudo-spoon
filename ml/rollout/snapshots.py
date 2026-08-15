"""Known-good config snapshots for production rollback (WS17).

A *snapshot* captures the per-flag state of a known-good rollout
configuration so that an operator can return to it in the event of a
production incident.  Snapshots are persisted as one JSON file per
snapshot under a configurable directory; the file name is the snapshot
name with a ``.json`` suffix.

The on-disk format is intentionally tiny and self-describing so the
operator can ``cat`` a snapshot during an incident::

    {
      "name": "pre-shrinkage-2026-08-12",
      "saved_at": "2026-08-12T11:14:02+00:00",
      "saved_by": "lbrunori",
      "commit_sha": "abc1234",
      "config_hash": "sha256:...",
      "flags": {
        "enable_shrinkage": {"stage": "disabled", "rollout_pct": 0.0},
        "enable_breakout_model": {"stage": "active", "rollout_pct": 25.0},
        ...
      }
    }

Design constraints (plan §19):

* **Idempotent** — re-saving a snapshot with the same name overwrites
  the previous one.  A re-load yields the same payload.
* **Auditable** — each snapshot carries ``saved_at``, ``saved_by``,
  ``commit_sha`` and ``config_hash`` so the chain of custody is clear.
* **Fast** — list/load operations are O(1)/O(n) on the snapshot
  directory and never touch the audit log.
* **Testable** — no I/O at import time; the directory is injected
  through :func:`save_snapshot` and :func:`load_snapshot`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config_hash import compute_config_hash

log = logging.getLogger(__name__)


# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable, JSON-serialisable known-good configuration snapshot.

    Attributes:
        name: Stable identifier (also the file basename, sans ``.json``).
        saved_at: ISO-8601 UTC timestamp at which the snapshot was saved.
        saved_by: Operator or system actor that captured the snapshot.
        commit_sha: Optional commit SHA at the time of capture.
        config_hash: Optional SHA-256 of the captured configuration.
        flags: Mapping ``flag_name → {stage, rollout_pct}`` representing
            the per-flag state at capture time.
    """

    name: str
    saved_at: str
    saved_by: str
    flags: Mapping[str, Mapping[str, Any]]
    commit_sha: str | None = None
    config_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    # ── Serialisation helpers ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "saved_at": self.saved_at,
            "saved_by": self.saved_by,
            "commit_sha": self.commit_sha,
            "config_hash": self.config_hash,
            "flags": {
                name: {"stage": f["stage"], "rollout_pct": float(f["rollout_pct"])}
                for name, f in self.flags.items()
            },
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Snapshot":
        raw_flags = payload.get("flags") or {}
        flags: dict[str, dict[str, Any]] = {}
        for flag_name, raw in raw_flags.items():
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"snapshot '{payload.get('name')}': flag '{flag_name}' "
                    "must be a mapping"
                )
            if "stage" not in raw:
                raise ValueError(
                    f"snapshot '{payload.get('name')}': flag '{flag_name}' "
                    "is missing 'stage'"
                )
            flags[str(flag_name)] = {
                "stage": str(raw["stage"]),
                "rollout_pct": float(raw.get("rollout_pct", 0.0)),
            }
        return cls(
            name=str(payload.get("name", "")),
            saved_at=str(payload.get("saved_at", "")),
            saved_by=str(payload.get("saved_by", "")),
            commit_sha=payload.get("commit_sha"),
            config_hash=payload.get("config_hash"),
            flags=flags,
            extra=dict(payload.get("extra") or {}),
        )


# ── File-system helpers ────────────────────────────────────────────────────


_SNAPSHOT_SUBDIR: str = "snapshots"
_SNAPSHOT_SUFFIX: str = ".json"
# Snapshot names: kebab-case / alnum + dash / underscore / dot, length 1-128.
_SNAPSHOT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class SnapshotError(ValueError):
    """Raised on snapshot validation / I/O failures."""


def snapshot_dir(artifacts_root: Path) -> Path:
    """Return the directory where snapshots are persisted."""
    return Path(artifacts_root) / _SNAPSHOT_SUBDIR


def snapshot_path(artifacts_root: Path, name: str) -> Path:
    """Return the file path for a given snapshot name."""
    _validate_name(name)
    return snapshot_dir(artifacts_root) / f"{name}{_SNAPSHOT_SUFFIX}"


def _validate_name(name: str) -> None:
    if not name or not isinstance(name, str):
        raise SnapshotError("snapshot name must be a non-empty string")
    if not _SNAPSHOT_NAME_RE.match(name):
        raise SnapshotError(
            f"invalid snapshot name {name!r}: allowed chars [A-Za-z0-9._-], "
            "length 1-128"
        )


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── Public API ─────────────────────────────────────────────────────────────


def save_snapshot(
    *,
    artifacts_root: Path,
    name: str,
    flags: Mapping[str, Mapping[str, Any]],
    saved_by: str,
    commit_sha: str | None = None,
    config_data: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    saved_at: str | None = None,
) -> Snapshot:
    """Persist a snapshot to disk (atomic write) and return it.

    The ``flags`` mapping is shallow-validated: every entry MUST have
    a ``stage`` key; ``rollout_pct`` defaults to ``0.0``.  When
    ``config_data`` is supplied, ``config_hash`` is computed and
    recorded so the snapshot ties back to a specific configuration
    bundle.

    Re-saving with the same ``name`` *overwrites* the previous
    snapshot.  This is intentional: keeping a single "current
    known-good" per name is easier to reason about during an incident
    than a history of superseded snapshots.

    The optional ``saved_at`` argument is for test injection: callers
    may pass a deterministic ISO-8601 timestamp to make snapshot
    ordering testable.  Production code should leave it ``None`` and
    rely on the auto-generated ``_utc_now_iso()``.
    """
    _validate_name(name)
    if not saved_by or not saved_by.strip():
        raise SnapshotError("saved_by must be a non-empty actor identifier")

    normalised_flags: dict[str, dict[str, Any]] = {}
    for flag_name, raw in flags.items():
        if not isinstance(raw, Mapping):
            raise SnapshotError(
                f"flag '{flag_name}' payload must be a mapping, got {type(raw).__name__}"
            )
        if "stage" not in raw:
            raise SnapshotError(
                f"flag '{flag_name}' is missing required 'stage' field"
            )
        normalised_flags[str(flag_name)] = {
            "stage": str(raw["stage"]),
            "rollout_pct": float(raw.get("rollout_pct", 0.0)),
        }

    config_hash: str | None = None
    if config_data is not None:
        config_hash = compute_config_hash(dict(config_data))

    snapshot = Snapshot(
        name=name,
        saved_at=saved_at or _utc_now_iso(),
        saved_by=saved_by,
        flags=normalised_flags,
        commit_sha=commit_sha,
        config_hash=config_hash,
        extra=dict(extra or {}),
    )

    target_dir = snapshot_dir(artifacts_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = snapshot_path(artifacts_root, name)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(snapshot.to_json(), encoding="utf-8")
    tmp.replace(target)
    log.info(
        "snapshot saved: name=%s flags=%d hash=%s → %s",
        name, len(normalised_flags), config_hash, target,
    )
    return snapshot


def load_snapshot(artifacts_root: Path, name: str) -> Snapshot:
    """Load a snapshot by name.  Raises :class:`SnapshotError` if absent."""
    _validate_name(name)
    path = snapshot_path(artifacts_root, name)
    if not path.is_file():
        raise SnapshotError(f"snapshot '{name}' not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"snapshot '{name}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(
            f"snapshot '{name}' payload must be a JSON object, got {type(payload).__name__}"
        )
    snapshot = Snapshot.from_dict(payload)
    if snapshot.name != name:
        raise SnapshotError(
            f"snapshot name mismatch: file {name!r} contains {snapshot.name!r}"
        )
    return snapshot


def list_snapshots(artifacts_root: Path) -> list[Snapshot]:
    """Return all snapshots, sorted by ``saved_at`` descending.

    Malformed snapshot files are skipped with a warning rather than
    aborting the listing — a corrupted file should not block
    observability of the rest of the directory.
    """
    target_dir = snapshot_dir(artifacts_root)
    if not target_dir.is_dir():
        return []
    out: list[Snapshot] = []
    for path in sorted(target_dir.glob(f"*{_SNAPSHOT_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.append(Snapshot.from_dict(payload))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            log.warning("skipping malformed snapshot %s: %s", path, exc)
    out.sort(key=lambda s: s.saved_at, reverse=True)
    return out


def latest_snapshot(artifacts_root: Path) -> Snapshot | None:
    """Return the most recent snapshot, or ``None`` if the directory is empty."""
    snapshots = list_snapshots(artifacts_root)
    return snapshots[0] if snapshots else None


def delete_snapshot(artifacts_root: Path, name: str) -> bool:
    """Remove a snapshot.  Returns ``True`` if a file was deleted.

    Idempotent: deleting a non-existent snapshot returns ``False``
    rather than raising.  This is intentional so that operators can
    safely run a cleanup script without bespoke existence checks.
    """
    _validate_name(name)
    path = snapshot_path(artifacts_root, name)
    if not path.is_file():
        return False
    path.unlink()
    log.info("snapshot deleted: name=%s path=%s", name, path)
    return True


__all__ = [
    "Snapshot",
    "SnapshotError",
    "snapshot_dir",
    "snapshot_path",
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "latest_snapshot",
    "delete_snapshot",
]
