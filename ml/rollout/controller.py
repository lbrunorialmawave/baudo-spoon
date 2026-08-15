"""Production rollout utilities (PR8 of the low-sample plan).

The rollout machinery sits *outside* the trainer: it consumes the
artefact produced by a Trainer run and decides, at inference time,
which scoring path to use.  The contract is intentionally small:

* :class:`FeatureFlag` — central registry of low-sample feature flags.
* :class:`ShadowModeRunner` — runs two scoring paths in parallel and
  logs the divergence.  No production impact.
* :class:`RolloutController` — decides, per-call, which path is
  authoritative.  Traffic is split by a configurable percentage.

Design principles (enforced by plan §48–55):

1. **Kill-switch by default.**  Every flag defaults to *off*, including
   the "use the new model" flag.  This keeps the production behaviour
   bit-for-bit identical until a PR5 experiment explicitly enables
   the path.
2. **Shadow first, then rollout.**  A flag must spend at least one
   monitoring cycle in ``SHADOW`` before it can be promoted to
   ``ACTIVE``.
3. **Atomic transitions.**  Promotion is a single call to
   :meth:`RolloutController.promote`; there is no in-between state
   that could be persisted accidentally.
4. **Auditable.**  Every transition is appended to an in-memory event
   log that callers can persist alongside the artefact.
5. **Gate-protected ACTIVE (WS6).**  Promotion to ``ACTIVE`` goes
   through :meth:`RolloutController.promote_to_active`, which runs the
   promotion gate and refuses the transition unless the gate passes.
   Break-glass is available for emergencies but must NEVER be used in
   the normal CI path (plan §8.5).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Mapping

from .audit import (
    AuditLog,
    record_denied,
    record_transition,
)
from .config_hash import compute_config_hash

log = logging.getLogger(__name__)


# ── Gate integration (WS6) ────────────────────────────────────────────────

# Re-export the gate types so callers can do
#   from ml.rollout import PromotionGateDenied
# without having to know which sub-module owns them.
from ml.scripts.check_promotion_gate import (  # noqa: E402
    PromotionGateDenied,
    PromotionGateError,
    PromotionGateReport,
    evaluate_report as _default_evaluate_report,
)

# A gate function is anything that takes a report path (and optionally
# a config snapshot) and returns a structured PromotionGateReport.
# We accept ``Callable[..., PromotionGateReport]`` so callers can use
# ``functools.partial`` to pre-bind thresholds.
GateFn = Callable[..., PromotionGateReport]


# ── Default rollout percentage ────────────────────────────────────────────

# Default traffic share used by ``promote_to_active`` when the caller
# does not pass ``new_rollout_pct`` explicitly.  Kept here (above
# ``RolloutController``) so the dataclass default can reference it.
DEFAULT_ROLLOUT_PCT: Final[float] = 10.0


# ── Feature flag definitions ─────────────────────────────────────────────────


class FeatureFlag(str, Enum):
    """Single source of truth for the low-sample feature flags.

    Each flag maps to a ``MLConfig`` field; the rollout controller
    reads from the live config to decide whether the path is active.
    """

    LIMITED_SAMPLE_TRAINING = "enable_limited_sample_training"
    PER90_SHRINKAGE = "enable_shrinkage"
    RECENT_ROLE_FEATURES = "enable_recent_role_features"
    BREAKOUT_MODEL = "enable_breakout_model"
    # String-valued MLConfig.reliability_weight_mode mapped onto the boolean
    # rollout machine: ACTIVE → "continuous", DISABLED/SHADOW → "bucket"
    # (plan-limited-cohort-patches.md G4 option a).
    RELIABILITY_WEIGHT_CONTINUOUS = "reliability_weight_mode"


# Stage of a feature flag.  Promotion must be monotonic.
class FlagStage(str, Enum):
    DISABLED = "disabled"   # off, no shadow
    SHADOW = "shadow"       # dual-scoring, no production impact
    ACTIVE = "active"       # authoritative for production traffic


# ── Shadow runner ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """Summary of a single shadow-mode scoring comparison.

    Attributes:
        n_rows: Number of rows scored.
        baseline_score: Aggregate score from the baseline (control) path.
        challenger_score: Aggregate score from the challenger path.
        absolute_delta: ``abs(challenger - baseline)``.
        relative_delta: ``absolute_delta / max(abs(baseline), eps)``.
        timestamp: UTC time at which the comparison was produced.
    """

    n_rows: int
    baseline_score: float
    challenger_score: float
    absolute_delta: float
    relative_delta: float
    timestamp: str


def shadow_compare(
    baseline_scores,
    challenger_scores,
    *,
    eps: float = 1e-9,
) -> ShadowComparison:
    """Compare two sequences of float scores and emit a ShadowComparison.

    The function is *pure* — it does not read config or call any
    estimator, and the only side effect is a single ``log.info`` line.
    This makes it easy to unit-test in isolation.
    """
    base_list = list(baseline_scores)
    chal_list = list(challenger_scores)
    if len(base_list) != len(chal_list):
        raise ValueError(
            f"Score sequences have different lengths "
            f"({len(base_list)} vs {len(chal_list)})"
        )
    n = len(base_list)
    if n == 0:
        baseline_mean = 0.0
        challenger_mean = 0.0
    else:
        baseline_mean = sum(base_list) / n
        challenger_mean = sum(chal_list) / n
    abs_delta = abs(challenger_mean - baseline_mean)
    rel_delta = abs_delta / max(abs(baseline_mean), eps)
    comparison = ShadowComparison(
        n_rows=n,
        baseline_score=baseline_mean,
        challenger_score=challenger_mean,
        absolute_delta=abs_delta,
        relative_delta=rel_delta,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
    log.info(
        "Shadow comparison: n=%d baseline=%.4f challenger=%.4f absΔ=%.4f relΔ=%.4f",
        comparison.n_rows, comparison.baseline_score,
        comparison.challenger_score, comparison.absolute_delta,
        comparison.relative_delta,
    )
    return comparison


# ── Rollout controller ──────────────────────────────────────────────────────


@dataclass
class RolloutController:
    """Decide which scoring path is authoritative for each request.

    The controller is **stateless across calls** (no I/O), but it keeps
    an in-memory event log of every transition so that operators can
    audit the rollout.  The event log is intentionally a plain list —
    persisting it is a caller concern.

    Attributes added in WS6:

    * ``audit_log`` — optional :class:`AuditLog` instance.  When
      provided, every successful and denied transition is recorded
      alongside the in-memory event list, with gate outcome and
      config hash.  When ``None``, only the in-memory event list is
      maintained (legacy behaviour).
    * ``gate_fn`` — optional callable used by
      :meth:`promote_to_active` to evaluate the promotion gate.  When
      ``None``, the controller defaults to
      :func:`ml.scripts.check_promotion_gate.evaluate_report`.  A
      custom gate function MUST be a callable that accepts
      ``report_path`` (and optionally ``config_snapshot``) and
      returns a :class:`PromotionGateReport`.
    """

    flag: FeatureFlag
    stage: FlagStage = FlagStage.DISABLED
    rollout_pct: float = 0.0
    random_seed: int = 0
    audit_log: AuditLog | None = None
    gate_fn: GateFn | None = None
    _events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rollout_pct <= 100.0:
            raise ValueError("rollout_pct must be in [0, 100]")
        if self.stage == FlagStage.ACTIVE and self.rollout_pct <= 0.0:
            raise ValueError("ACTIVE stage requires rollout_pct > 0")
        self._rng = random.Random(self.random_seed)
        if self.gate_fn is None:
            self.gate_fn = _default_evaluate_report

    def is_active(self) -> bool:
        return self.stage == FlagStage.ACTIVE

    def use_challenger(self) -> bool:
        """Return True if the *challenger* path must be used for this call.

        Behaviour by stage:

        * ``DISABLED`` → always ``False``.
        * ``SHADOW``  → always ``False`` (the challenger is only
          observed, never authoritative).
        * ``ACTIVE``  → Bernoulli sample with probability
          ``rollout_pct / 100``.
        """
        if self.stage == FlagStage.DISABLED:
            return False
        if self.stage == FlagStage.SHADOW:
            return False
        return self._rng.random() * 100.0 < self.rollout_pct

    def promote(
        self,
        *,
        new_stage: FlagStage,
        new_rollout_pct: float | None = None,
        config_snapshot: Mapping[str, object] | None = None,
    ) -> None:
        """Atomically transition to a new stage.

        Promotion must be monotonic (``DISABLED → SHADOW → ACTIVE``);
        a rewind is allowed (e.g. for an emergency rollback) but emits
        an ``"emergency_rollback"`` reason.

        If ``config_snapshot`` is provided, its canonical SHA-256
        (``config_hash``, plan §18) is recorded on the transition
        event.  This makes the audit log self-describing: every stage
        change carries the exact configuration that produced it.

        Operational gate (plan-limited-cohort-hardening WS4 — manual,
        not enforced in this method which stays I/O-free by design):

        Before promoting any low-sample flag family member
        (``LIMITED_SAMPLE_TRAINING``, ``PER90_SHRINKAGE``, …) to
        ``ACTIVE``, the operator must attach:

        1. An experiment-harness report containing cohort-stratified
           metrics (``mae_by_cohort``, ``rmse_by_cohort``,
           ``phenom_leakage_rate``).
        2. Evidence that known canary anomalies (Adzic-class) are
           resolved (0 residual anomalies in the top bracket).

        Prefer promoting ``PER90_SHRINKAGE`` together with (or before)
        leaving ``LIMITED_SAMPLE_TRAINING`` at full traffic, so the
        protective path is never missing while LIMITED rows influence
        the model.
        """
        if new_stage not in FlagStage:
            raise ValueError(f"Unknown stage: {new_stage!r}")
        order = [FlagStage.DISABLED, FlagStage.SHADOW, FlagStage.ACTIVE]
        if order.index(new_stage) < order.index(self.stage):
            reason = "emergency_rollback"
        else:
            reason = "promotion"
        event = {
            "flag": self.flag.value,
            "from_stage": self.stage.value,
            "to_stage": new_stage.value,
            "from_pct": self.rollout_pct,
            "to_pct": new_rollout_pct if new_rollout_pct is not None else self.rollout_pct,
            "reason": reason,
            "at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if config_snapshot is not None:
            event["config_hash"] = compute_config_hash(config_snapshot)
        self.stage = new_stage
        if new_rollout_pct is not None:
            if not 0.0 <= new_rollout_pct <= 100.0:
                raise ValueError("new_rollout_pct must be in [0, 100]")
            self.rollout_pct = float(new_rollout_pct)
        self._events.append(event)
        log.info(
            "RolloutController[%s] %s → %s (%.1f%%) reason=%s",
            self.flag.value, event["from_stage"], event["to_stage"],
            self.rollout_pct, reason,
        )

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    # ── Gate-protected ACTIVE transition (WS6) ────────────────────────────

    def promote_to_active(
        self,
        *,
        report_path: Path | str,
        config_snapshot: Mapping[str, object] | None = None,
        config_snapshot_path: Path | str | None = None,
        actor: str = "unknown",
        commit_sha: str | None = None,
        new_rollout_pct: float = DEFAULT_ROLLOUT_PCT,
        break_glass: bool = False,
        break_glass_reason: str | None = None,
    ) -> PromotionGateReport:
        """Atomically transition to ``ACTIVE`` iff the promotion gate PASSES.

        Plan §8 — "Hard promotion gate" — forbids promoting a
        low-sample flag to ``ACTIVE`` without an experiment-harness
        report whose cohort-aware metrics, canary status, and
        config-hash cross-check all pass.

        Behaviour:

        1. The configured ``gate_fn`` (default:
           :func:`ml.scripts.check_promotion_gate.evaluate_report`)
           evaluates the report and returns a
           :class:`PromotionGateReport`.
        2. If ``passed`` is True → the transition happens through
           :meth:`promote` and the in-memory event carries
           ``gate_result="PASS"`` and the candidate ``config_hash``.
           The audit log (when present) receives a
           ``TRANSITION`` record with ``gate_result="PASS"``.
        3. If ``passed`` is False and ``break_glass`` is False →
           :class:`PromotionGateDenied` is raised.  No state change
           occurs.  When an audit log is attached, a ``DENIED``
           record is appended with the full list of failures.
        4. If ``passed`` is False and ``break_glass`` is True →
           ``break_glass_reason`` MUST be supplied (otherwise
           :class:`ValueError` is raised).  The transition proceeds
           with ``reason="break_glass"``; the audit log records the
           break-glass event with the reason.

        Break-glass is **NEVER** meant to be set in the normal CI
        promotion workflow (plan §8.5).  It exists solely for an
        operator to override the gate during an incident, with full
        audit trail.

        Returns the :class:`PromotionGateReport` (whether the gate
        passed or was overridden).
        """
        if break_glass and not (break_glass_reason and break_glass_reason.strip()):
            raise ValueError(
                "break_glass=True requires a non-empty break_glass_reason"
            )

        report_path = Path(report_path)
        snapshot_path: Path | None = None
        snapshot_map: Mapping[str, object] | None = config_snapshot
        if config_snapshot_path is not None:
            snapshot_path = Path(config_snapshot_path)

        # ── Idempotency guard (WS16, plan §18) ──────────────────────────
        # Re-running the rollout workflow against an already-ACTIVE flag
        # MUST be a no-op when the configuration is unchanged.  This
        # keeps the pipeline idempotent (a second ``ml-training.yml``
        # run must not flip the flag back to ACTIVE through a stale
        # report downloaded from R2).  We compare the live
        # ``config_hash`` recorded on the most recent successful
        # transition against the candidate's hash; only when they
        # match do we short-circuit.
        if (
            self.stage == FlagStage.ACTIVE
            and not break_glass
        ):
            last_event = self._events[-1] if self._events else None
            last_hash = (
                last_event.get("config_hash") if last_event else None
            )
            candidate_hash: str | None = None
            if snapshot_map is not None:
                candidate_hash = compute_config_hash(snapshot_map)
            elif snapshot_path is not None:
                # Re-hash the on-disk snapshot so callers that only
                # pass a path still benefit from the early-return.
                try:
                    import json as _json
                    candidate_hash = compute_config_hash(
                        _json.loads(snapshot_path.read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError):
                    candidate_hash = None
            if (
                candidate_hash is not None
                and last_hash is not None
                and last_hash == candidate_hash
            ):
                log.info(
                    "RolloutController[%s] already ACTIVE with the same "
                    "config_hash=%s — idempotent replay, skipping gate",
                    self.flag.value,
                    candidate_hash,
                )
                return PromotionGateReport(
                    passed=True,
                    failures=(),
                    report_path=str(report_path),
                    variant="?",
                    config_hash=candidate_hash,
                    config_hash_status="match",
                    extra={"idempotent_replay": True},
                )

        # The default gate function accepts (report_path, *, config_snapshot=...).
        # It also accepts the legacy positional variant; we pass the path
        # as a keyword to be unambiguous.
        if self.gate_fn is None:
            # Should not happen because __post_init__ defaults it, but be
            # explicit: fail-closed means no gate = no ACTIVE.
            raise PromotionGateError(
                "RolloutController.gate_fn is None — refuse to promote to ACTIVE"
            )

        try:
            outcome = self.gate_fn(
                report_path,
                config_snapshot=snapshot_path,
            )
        except (OSError, ValueError) as exc:
            # I/O or schema errors are treated as a hard gate failure.
            denial = PromotionGateReport(
                passed=False,
                failures=(f"gate evaluation error: {exc!r}",),
                report_path=str(report_path),
                variant="?",
            )
            self._record_promotion_outcome(
                outcome=denial,
                actor=actor,
                commit_sha=commit_sha,
                config_snapshot=snapshot_map,
                config_snapshot_path=snapshot_path,
                target_stage=FlagStage.ACTIVE,
            )
            raise PromotionGateDenied(
                f"promotion gate could not be evaluated: {exc!r}",
                denial,
            ) from exc

        # Compute the candidate config_hash for audit, even if no snapshot
        # was supplied to the gate (e.g. tests that don't pass one).
        candidate_hash: str | None = None
        if snapshot_map is not None:
            candidate_hash = compute_config_hash(snapshot_map)
        elif outcome.config_hash is not None:
            candidate_hash = outcome.config_hash

        if not outcome.passed:
            if not break_glass:
                self._record_promotion_outcome(
                    outcome=outcome,
                    actor=actor,
                    commit_sha=commit_sha,
                    config_snapshot=snapshot_map,
                    config_snapshot_path=snapshot_path,
                    target_stage=FlagStage.ACTIVE,
                )
                raise PromotionGateDenied(
                    f"promotion gate FAILED for flag={self.flag.value!r}: "
                    f"{len(outcome.failures)} check(s) failed",
                    outcome,
                )
            # Break-glass path: record the override and proceed.
            log.warning(
                "RolloutController[%s] BREAK-GLASS override by actor=%s "
                "reason=%r despite %d gate failure(s)",
                self.flag.value, actor, break_glass_reason, len(outcome.failures),
            )

        # Transition (gate PASS or break-glass override).
        from_stage = self.stage
        from_pct = self.rollout_pct
        self.promote(
            new_stage=FlagStage.ACTIVE,
            new_rollout_pct=new_rollout_pct,
            config_snapshot=snapshot_map,
        )
        # Annotate the most recent event with gate outcome.
        if self._events:
            ev = self._events[-1]
            ev["gate_result"] = "BREAK_GLASS" if break_glass else "PASS"
            ev["gate_failures"] = list(outcome.failures)
            if candidate_hash is not None:
                ev["config_hash"] = candidate_hash
            ev["report_path"] = str(report_path)
            ev["actor"] = actor
            ev["commit_sha"] = commit_sha
            if break_glass:
                ev["reason"] = "break_glass"
                ev["break_glass_reason"] = break_glass_reason
            ev["from_stage"] = from_stage.value
            ev["from_pct"] = from_pct

        if self.audit_log is not None:
            extra: dict[str, Any] = {
                "report_path": str(report_path),
                "variant": outcome.variant,
                "config_hash_status": outcome.config_hash_status,
            }
            if break_glass:
                extra["break_glass_reason"] = break_glass_reason
                extra["gate_failures_at_override"] = list(outcome.failures)
            reason_text = (
                "break_glass" if break_glass else "promotion_gate_passed"
            )
            self.audit_log.append(
                record_transition(
                    actor=actor,
                    flag=self.flag.value,
                    from_stage=from_stage.value,
                    to_stage=FlagStage.ACTIVE.value,
                    from_pct=from_pct,
                    to_pct=self.rollout_pct,
                    reason=reason_text,
                    commit_sha=commit_sha,
                    promotion_report=str(report_path),
                    gate_result="BREAK_GLASS" if break_glass else "PASS",
                    config_hash=candidate_hash,
                    extra=extra,
                )
            )
        return outcome

    def _record_promotion_outcome(
        self,
        *,
        outcome: PromotionGateReport,
        actor: str,
        commit_sha: str | None,
        config_snapshot: Mapping[str, object] | None,
        config_snapshot_path: Path | None,
        target_stage: FlagStage,
    ) -> None:
        """Append a ``DENIED`` audit record (when an audit log is attached)."""
        if self.audit_log is None:
            return
        candidate_hash: str | None = None
        if config_snapshot is not None:
            candidate_hash = compute_config_hash(config_snapshot)
        elif outcome.config_hash is not None:
            candidate_hash = outcome.config_hash
        self.audit_log.append(
            record_denied(
                actor=actor,
                attempted_from=self.stage.value,
                attempted_to=target_stage.value,
                reason="promotion_gate_failed",
                failed_checks=outcome.failures,
                commit_sha=commit_sha,
                config_hash=candidate_hash,
                flag=self.flag.value,
                extra={
                    "report_path": outcome.report_path,
                    "variant": outcome.variant,
                    "config_hash_status": outcome.config_hash_status,
                    "config_snapshot_path": str(config_snapshot_path)
                    if config_snapshot_path is not None
                    else None,
                },
            )
        )


# ── Default controllers ─────────────────────────────────────────────────────


def default_controllers(*, random_seed: int = 0) -> dict[FeatureFlag, RolloutController]:
    """Return a fresh, all-disabled set of rollout controllers.

    The set is the canonical "production" starting point: every flag
    sits at ``DISABLED`` with ``rollout_pct=0``.  Callers can promote
    individual flags to ``SHADOW`` or ``ACTIVE`` explicitly.
    """
    return {
        flag: RolloutController(flag=flag, random_seed=random_seed)
        for flag in FeatureFlag
    }


def reliability_weight_mode_for_stage(stage: FlagStage) -> str:
    """Map rollout stage → reliability_weight_mode string.

    ACTIVE → continuous; DISABLED or SHADOW → bucket (safe / legacy path
    while the continuous mode is still being observed in shadow).
    """
    if stage == FlagStage.ACTIVE:
        return "continuous"
    return "bucket"
