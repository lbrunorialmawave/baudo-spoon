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
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Final

log = logging.getLogger(__name__)


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
    """

    flag: FeatureFlag
    stage: FlagStage = FlagStage.DISABLED
    rollout_pct: float = 0.0
    random_seed: int = 0
    _events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rollout_pct <= 100.0:
            raise ValueError("rollout_pct must be in [0, 100]")
        if self.stage == FlagStage.ACTIVE and self.rollout_pct <= 0.0:
            raise ValueError("ACTIVE stage requires rollout_pct > 0")
        self._rng = random.Random(self.random_seed)

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

    def promote(self, *, new_stage: FlagStage, new_rollout_pct: float | None = None) -> None:
        """Atomically transition to a new stage.

        Promotion must be monotonic (``DISABLED → SHADOW → ACTIVE``);
        a rewind is allowed (e.g. for an emergency rollback) but emits
        an ``"emergency_rollback"`` reason.

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


# ── Default controllers ─────────────────────────────────────────────────────

DEFAULT_ROLLOUT_PCT: Final[float] = 10.0


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
