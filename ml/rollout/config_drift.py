"""Configuration drift detector (WS5 of plan.md).

Detects divergence between:

* the **declared** rollout state (DISABLED / SHADOW / ACTIVE) and
* the **effective** runtime configuration actually observed.

Drift is classified by type and severity so that CI / promotion gates
can act on it deterministically:

* ``P0`` — production behaviour mismatch.  Any ``P0`` finding makes
  the report fail-closed (``exit 1``).  Promotion to ``ACTIVE`` is
  forbidden while any P0 is open.
* ``P1`` — challenger / metadata mismatch (e.g. SHADOW declared but
  no challenger env var set).  Reported; does not fail CI by default
  but is recorded in the promotion report.
* ``P2`` — informational only.

The detector is pure (no I/O): the caller composes the inputs.  A
small CLI wrapper (:mod:`ml.scripts.check_config_drift`) executes it
against the live environment for CI usage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from .controller import FeatureFlag, FlagStage
from .env_flags import ResolvedFlags

log = logging.getLogger(__name__)


# ── Drift taxonomy ───────────────────────────────────────────────────────────


class DriftType(str, Enum):
    """Closed set of drift categories the detector emits."""

    MODE_MISMATCH = "mode_mismatch"
    BOOLEAN_MISMATCH = "boolean_mismatch"
    STAGE_MISMATCH = "stage_mismatch"
    CHALLENGER_MISMATCH = "challenger_mismatch"
    MISSING_FIELD = "missing_field"
    INVALID_VALUE = "invalid_value"


class DriftSeverity(str, Enum):
    """Severity ladder — ordered from highest to lowest."""

    P0 = "P0"  # production behaviour mismatch (fail-closed)
    P1 = "P1"  # challenger / metadata mismatch
    P2 = "P2"  # informational


_SEVERITY_ORDER: tuple[DriftSeverity, ...] = (
    DriftSeverity.P0,
    DriftSeverity.P1,
    DriftSeverity.P2,
)


# Per-drift-type severity.  Centralised so that no other module has to
# reason about which drift is blocking.
_DEFAULT_SEVERITY: dict[DriftType, DriftSeverity] = {
    DriftType.MODE_MISMATCH: DriftSeverity.P0,
    DriftType.BOOLEAN_MISMATCH: DriftSeverity.P0,
    DriftType.STAGE_MISMATCH: DriftSeverity.P0,
    DriftType.CHALLENGER_MISMATCH: DriftSeverity.P1,
    DriftType.MISSING_FIELD: DriftSeverity.P1,
    DriftType.INVALID_VALUE: DriftSeverity.P0,
}


# ── Snapshot types ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RolloutSnapshot:
    """Declared rollout state — what the *operator* says the system is.

    Constructed from a :class:`RolloutController` (preferred) or
    directly from a deployment manifest / promotion report.
    """

    stage: FlagStage
    production_mode: str  # "bucket" | "continuous"
    production_flags: Mapping[str, bool] = field(default_factory=dict)
    challenger_flags: Mapping[str, bool] = field(default_factory=dict)
    source: str = "unspecified"

    def __post_init__(self) -> None:
        mode = (self.production_mode or "").strip()
        if mode not in {"bucket", "continuous"}:
            raise ValueError(
                f"RolloutSnapshot.production_mode must be bucket|continuous, "
                f"got {self.production_mode!r}"
            )
        # Store canonical form (no silent case-folding — the user must
        # pass the exact literal to avoid surprises).
        if self.production_mode != mode:
            object.__setattr__(self, "production_mode", mode)
        if not isinstance(self.stage, FlagStage):
            raise ValueError(
                f"RolloutSnapshot.stage must be FlagStage, got {type(self.stage).__name__}"
            )


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    """Effective runtime config — what the system is *actually* doing.

    Built from the live :class:`MLConfig`, the :class:`DataRepository`
    instantiation params, or any consumer that has resolved its flags
    via :func:`ml.rollout.env_flags.resolve_env_flags`.
    """

    production_mode: str
    use_new_behavior: bool
    production_flags: Mapping[str, bool] = field(default_factory=dict)
    challenger_enabled: bool = False
    source: str = "unspecified"

    def __post_init__(self) -> None:
        mode = (self.production_mode or "").strip()
        if mode not in {"bucket", "continuous"}:
            raise ValueError(
                f"EffectiveConfig.production_mode must be bucket|continuous, "
                f"got {self.production_mode!r}"
            )
        if self.production_mode != mode:
            object.__setattr__(self, "production_mode", mode)


# ── Findings + report ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DriftFinding:
    """A single drift observation."""

    drift_type: DriftType
    severity: DriftSeverity
    field: str
    expected: Any
    actual: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Aggregate result of a drift detection pass."""

    findings: tuple[DriftFinding, ...] = field(default_factory=tuple)
    rollout_stage: str = ""
    effective_mode: str = ""
    rollout_source: str = ""
    effective_source: str = ""
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def has_drift(self) -> bool:
        return len(self.findings) > 0

    @property
    def has_p0(self) -> bool:
        return any(f.severity == DriftSeverity.P0 for f in self.findings)

    @property
    def has_p1(self) -> bool:
        return any(f.severity == DriftSeverity.P1 for f in self.findings)

    @property
    def highest_severity(self) -> DriftSeverity | None:
        if not self.findings:
            return None
        severities = {f.severity for f in self.findings}
        for s in _SEVERITY_ORDER:
            if s in severities:
                return s
        return None

    def exit_code(self) -> int:
        """Return 1 if any P0 is present, else 0.

        The promotion gate treats any non-zero as ``DENY`` (fail-closed).
        """
        return 1 if self.has_p0 else 0

    def by_severity(self, severity: DriftSeverity) -> list[DriftFinding]:
        return [f for f in self.findings if f.severity == severity]

    def by_type(self, drift_type: DriftType) -> list[DriftFinding]:
        return [f for f in self.findings if f.drift_type == drift_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "rollout_stage": self.rollout_stage,
            "effective_mode": self.effective_mode,
            "rollout_source": self.rollout_source,
            "effective_source": self.effective_source,
            "generated_at_utc": self.generated_at_utc,
            "exit_code": self.exit_code(),
            "highest_severity": (
                self.highest_severity.value if self.highest_severity else None
            ),
            "summary": {
                "total": len(self.findings),
                "p0": len(self.by_severity(DriftSeverity.P0)),
                "p1": len(self.by_severity(DriftSeverity.P1)),
                "p2": len(self.by_severity(DriftSeverity.P2)),
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── Detection logic ─────────────────────────────────────────────────────────


def _emit(
    findings: list[DriftFinding],
    drift_type: DriftType,
    field: str,
    expected: Any,
    actual: Any,
    message: str,
) -> None:
    findings.append(
        DriftFinding(
            drift_type=drift_type,
            severity=_DEFAULT_SEVERITY[drift_type],
            field=field,
            expected=expected,
            actual=actual,
            message=message,
        )
    )


def _validate_mode_field(value: Any, *, field: str) -> str | None:
    """Return mode if value is exactly ``bucket`` or ``continuous``, else None."""
    if not isinstance(value, str):
        return None
    mode = value.strip()
    return mode if mode in {"bucket", "continuous"} else None


def detect_config_drift(
    rollout: RolloutSnapshot,
    effective: EffectiveConfig,
) -> DriftReport:
    """Compare declared rollout state vs effective runtime config.

    The detector never raises; it always returns a :class:`DriftReport`.
    Callers inspect ``report.exit_code()`` to decide promotion.
    """
    findings: list[DriftFinding] = []

    # 1. Mode validity (INVALID_VALUE → P0)
    rollout_mode = _validate_mode_field(
        rollout.production_mode, field="rollout.production_mode"
    )
    if rollout_mode is None:
        _emit(
            findings,
            DriftType.INVALID_VALUE,
            "rollout.production_mode",
            expected="bucket|continuous",
            actual=rollout.production_mode,
            message=(
                f"RolloutSnapshot.production_mode={rollout.production_mode!r} is not "
                "one of {bucket, continuous}"
            ),
        )
    effective_mode = _validate_mode_field(
        effective.production_mode, field="effective.production_mode"
    )
    if effective_mode is None:
        _emit(
            findings,
            DriftType.INVALID_VALUE,
            "effective.production_mode",
            expected="bucket|continuous",
            actual=effective.production_mode,
            message=(
                f"EffectiveConfig.production_mode={effective.production_mode!r} is not "
                "one of {bucket, continuous}"
            ),
        )

    # 2. Stage / use_new_behavior coherence (STAGE_MISMATCH → P0)
    expected_use_new = rollout.stage == FlagStage.ACTIVE
    if effective.use_new_behavior != expected_use_new:
        _emit(
            findings,
            DriftType.STAGE_MISMATCH,
            "effective.use_new_behavior",
            expected=expected_use_new,
            actual=effective.use_new_behavior,
            message=(
                f"Rollout stage={rollout.stage.value!r} implies "
                f"use_new_behavior={expected_use_new} but effective reports "
                f"{effective.use_new_behavior}"
            ),
        )

    # 3. Mode coherence (MODE_MISMATCH → P0, unless it's the SHADOW case)
    if rollout_mode is not None and effective_mode is not None:
        if rollout_mode != effective_mode:
            # In SHADOW, production must stay bucket while challenger
            # observes continuous → expected mismatch.  Anything else is P0.
            if rollout.stage == FlagStage.SHADOW and effective_mode == "bucket":
                # Expected production-mode=bucket while in SHADOW.
                pass
            else:
                _emit(
                    findings,
                    DriftType.MODE_MISMATCH,
                    "production_mode",
                    expected=rollout_mode,
                    actual=effective_mode,
                    message=(
                        f"Rollout mode={rollout_mode!r} but effective mode="
                        f"{effective_mode!r}"
                    ),
                )

    # 4. Boolean flag parity (BOOLEAN_MISMATCH → P0)
    for flag_name, declared in rollout.production_flags.items():
        actual = effective.production_flags.get(flag_name)
        if actual is None:
            _emit(
                findings,
                DriftType.MISSING_FIELD,
                f"effective.production_flags[{flag_name!r}]",
                expected=declared,
                actual=None,
                message=(
                    f"Flag {flag_name!r} declared by rollout "
                    f"({declared}) but missing from effective config"
                ),
            )
            continue
        if bool(actual) != bool(declared):
            _emit(
                findings,
                DriftType.BOOLEAN_MISMATCH,
                f"effective.production_flags[{flag_name!r}]",
                expected=bool(declared),
                actual=bool(actual),
                message=(
                    f"Flag {flag_name!r}: rollout={declared} but effective={actual}"
                ),
            )

    # 5. Challenger coherence (CHALLENGER_MISMATCH → P1)
    if rollout.stage == FlagStage.SHADOW and not effective.challenger_enabled:
        _emit(
            findings,
            DriftType.CHALLENGER_MISMATCH,
            "effective.challenger_enabled",
            expected=True,
            actual=False,
            message="Rollout stage=SHADOW but effective.challenger_enabled=False",
        )
    if rollout.stage == FlagStage.DISABLED and effective.challenger_enabled:
        _emit(
            findings,
            DriftType.CHALLENGER_MISMATCH,
            "effective.challenger_enabled",
            expected=False,
            actual=True,
            message="Rollout stage=DISABLED but effective.challenger_enabled=True",
        )
    # NOTE: ACTIVE without challenger_enabled is *not* a finding — once a
    # flag reaches ACTIVE the challenger observation window is closed.
    # Observability of "challenger was ever present" is logged elsewhere.

    log.info(
        "Config drift: %d finding(s) (highest=%s)",
        len(findings),
        _SEVERITY_ORDER[0].value if any(
            f.severity == _SEVERITY_ORDER[0] for f in findings
        ) else "none",
    )
    return DriftReport(
        findings=tuple(findings),
        rollout_stage=rollout.stage.value,
        effective_mode=effective_mode or "",
        rollout_source=rollout.source,
        effective_source=effective.source,
    )


# ── Helpers to build snapshots from existing types ───────────────────────────


def rollout_snapshot_from_resolved(
    resolved: ResolvedFlags,
    *,
    source: str = "env_flags",
) -> RolloutSnapshot:
    """Build a :class:`RolloutSnapshot` from a :class:`ResolvedFlags`.

    The "stage" is derived as the union of per-flag stages: if any
    flag is ``ACTIVE`` we report ``ACTIVE``; else if any is ``SHADOW``
    we report ``SHADOW``; otherwise ``DISABLED``.  This mirrors how a
    single deployment manifest declares a stage.
    """
    if any(stage == FlagStage.ACTIVE.value for stage in resolved.stages.values()):
        stage = FlagStage.ACTIVE
    elif any(stage == FlagStage.SHADOW.value for stage in resolved.stages.values()):
        stage = FlagStage.SHADOW
    else:
        stage = FlagStage.DISABLED

    production_mode = (
        "continuous" if stage == FlagStage.ACTIVE else "bucket"
    )
    return RolloutSnapshot(
        stage=stage,
        production_mode=production_mode,
        production_flags=dict(resolved.production),
        challenger_flags=dict(resolved.challenger),
        source=source,
    )


def effective_config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source: str = "unspecified",
) -> EffectiveConfig:
    """Build :class:`EffectiveConfig` from a generic mapping.

    Recognised keys:

    * ``production_mode`` / ``reliability_weight_mode``
    * ``use_new_behavior`` / ``compute_new_behavior`` (fallback if no
      ``use_new_behavior``)
    * ``production_flags`` (dict[str, bool])
    * ``challenger_enabled``
    """
    mode = (
        mapping.get("production_mode")
        or mapping.get("reliability_weight_mode")
        or "bucket"
    )
    use_new = mapping.get("use_new_behavior")
    if use_new is None:
        use_new = bool(mapping.get("compute_new_behavior", False))
    production_flags = dict(mapping.get("production_flags") or {})
    challenger_enabled = bool(mapping.get("challenger_enabled", False))
    return EffectiveConfig(
        production_mode=str(mode),
        use_new_behavior=bool(use_new),
        production_flags=production_flags,
        challenger_enabled=challenger_enabled,
        source=source,
    )


def merge_reports(reports: Iterable[DriftReport]) -> DriftReport:
    """Merge multiple :class:`DriftReport` instances.

    Used when comparing against multiple effective sources (e.g. one
    snapshot per consumer).  Highest-severity findings win.
    """
    merged: list[DriftFinding] = []
    rollout_stage = ""
    effective_mode = ""
    rollout_source = ""
    effective_source = ""
    for r in reports:
        merged.extend(r.findings)
        if not rollout_stage and r.rollout_stage:
            rollout_stage = r.rollout_stage
        if not effective_mode and r.effective_mode:
            effective_mode = r.effective_mode
        if not rollout_source and r.rollout_source:
            rollout_source = r.rollout_source
        if not effective_source and r.effective_source:
            effective_source = r.effective_source

    # Stable order: P0 first, then P1, then P2; insertion-stable within tier.
    tier = {DriftSeverity.P0: 0, DriftSeverity.P1: 1, DriftSeverity.P2: 2}
    merged.sort(key=lambda f: (tier[f.severity], f.drift_type.value, f.field))

    return DriftReport(
        findings=tuple(merged),
        rollout_stage=rollout_stage,
        effective_mode=effective_mode,
        rollout_source=rollout_source,
        effective_source=effective_source,
    )


def render_markdown(report: DriftReport) -> str:
    """Return a human-readable Markdown summary (logs, PR comments)."""
    lines: list[str] = []
    lines.append("# Configuration drift report")
    lines.append("")
    lines.append(f"- **Generated:** {report.generated_at_utc}")
    lines.append(f"- **Rollout stage:** `{report.rollout_stage or '?'}`")
    lines.append(f"- **Effective mode:** `{report.effective_mode or '?'}`")
    lines.append(f"- **Rollout source:** `{report.rollout_source or '?'}`")
    lines.append(f"- **Effective source:** `{report.effective_source or '?'}`")
    lines.append(f"- **Highest severity:** `{report.highest_severity.value if report.highest_severity else 'NONE'}`")
    lines.append(f"- **Exit code:** `{report.exit_code()}`")
    lines.append("")
    summary = {
        "P0": len(report.by_severity(DriftSeverity.P0)),
        "P1": len(report.by_severity(DriftSeverity.P1)),
        "P2": len(report.by_severity(DriftSeverity.P2)),
    }
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- P0: {summary['P0']}")
    lines.append(f"- P1: {summary['P1']}")
    lines.append(f"- P2: {summary['P2']}")
    lines.append("")
    if not report.findings:
        lines.append("No drift detected. ✅")
        return "\n".join(lines)

    lines.append("## Findings")
    lines.append("")
    lines.append("| Severity | Type | Field | Expected | Actual | Message |")
    lines.append("|---|---|---|---|---|---|")
    for f in report.findings:
        exp = json.dumps(f.expected, ensure_ascii=False)
        act = json.dumps(f.actual, ensure_ascii=False)
        msg = f.message.replace("|", "\\|")
        lines.append(
            f"| {f.severity.value} | {f.drift_type.value} | "
            f"`{f.field}` | `{exp}` | `{act}` | {msg} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DriftType",
    "DriftSeverity",
    "DriftFinding",
    "DriftReport",
    "RolloutSnapshot",
    "EffectiveConfig",
    "detect_config_drift",
    "rollout_snapshot_from_resolved",
    "effective_config_from_mapping",
    "merge_reports",
    "render_markdown",
]
