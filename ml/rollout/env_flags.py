"""Resolve production vs challenger feature flags from environment (WS3).

The CI workflow (ml-training.yml) exports:

* ``ML_<FLAG>=true``              → production path uses the challenger (ACTIVE)
* ``ML_<FLAG>_CHALLENGER=true``   → compute/observe only (SHADOW); production stays legacy

This module is the single place that interprets those env vars so Trainer,
harness, and deployment contract stay aligned.

Boolean flags covered (FeatureFlag values):
  enable_limited_sample_training
  enable_shrinkage
  enable_recent_role_features
  enable_breakout_model

String modes:
  ML_RELIABILITY_WEIGHT_MODE=bucket|continuous
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

from .controller import FeatureFlag, FlagStage, reliability_weight_mode_for_stage

# Map FeatureFlag enum value → env suffix (upper snake without ML_ prefix)
_BOOL_FLAGS: Final[tuple[str, ...]] = tuple(f.value for f in FeatureFlag if f != FeatureFlag.RELIABILITY_WEIGHT_CONTINUOUS)

_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUE


@dataclass(frozen=True, slots=True)
class ResolvedFlags:
    """Effective production flags + challenger (shadow) flags."""

    production: dict[str, bool] = field(default_factory=dict)
    challenger: dict[str, bool] = field(default_factory=dict)
    reliability_weight_mode: str = "bucket"
    stages: dict[str, str] = field(default_factory=dict)

    def any_challenger(self) -> bool:
        return any(self.challenger.values())

    def stage_for(self, flag_name: str) -> str:
        return self.stages.get(flag_name, FlagStage.DISABLED.value)


def resolve_env_flags(
    *,
    env: dict[str, str] | None = None,
) -> ResolvedFlags:
    """Parse env into production vs challenger flag sets.

    Priority per flag:
      ACTIVE (ML_<FLAG>=true)  → production=True, challenger=True, stage=active
      SHADOW (ML_<FLAG>_CHALLENGER=true only) → production=False, challenger=True, stage=shadow
      else → both False, stage=disabled

    ``reliability_weight_mode``:
      Explicit ML_RELIABILITY_WEIGHT_MODE wins if valid.
      Else derived from continuous-flag stage via reliability_weight_mode_for_stage.
    """
    source = env if env is not None else dict(os.environ)

    production: dict[str, bool] = {}
    challenger: dict[str, bool] = {}
    stages: dict[str, str] = {}

    for name in _BOOL_FLAGS:
        upper = name.upper()
        active = _truthy(source.get(f"ML_{upper}"))
        shadow = _truthy(source.get(f"ML_{upper}_CHALLENGER"))
        if active:
            production[name] = True
            challenger[name] = True
            stages[name] = FlagStage.ACTIVE.value
        elif shadow:
            production[name] = False
            challenger[name] = True
            stages[name] = FlagStage.SHADOW.value
        else:
            production[name] = False
            challenger[name] = False
            stages[name] = FlagStage.DISABLED.value

    # reliability_weight_mode
    mode_raw = source.get("ML_RELIABILITY_WEIGHT_MODE")
    if mode_raw is not None and mode_raw.strip().lower() in {"bucket", "continuous"}:
        mode = mode_raw.strip().lower()
    else:
        # Derive from continuous flag stage if present
        cont_stage = stages.get(FeatureFlag.RELIABILITY_WEIGHT_CONTINUOUS.value)
        # The continuous flag is string-valued; map via ACTIVE env of the bool-ish name
        # or explicit stage for enable_shrinkage family default.
        # Prefer: if any production low-sample flag is active → continuous else bucket
        if any(production.values()):
            mode = reliability_weight_mode_for_stage(FlagStage.ACTIVE)
        elif any(challenger.values()):
            mode = reliability_weight_mode_for_stage(FlagStage.SHADOW)
        else:
            mode = reliability_weight_mode_for_stage(FlagStage.DISABLED)
        if cont_stage == FlagStage.ACTIVE.value:
            mode = "continuous"

    return ResolvedFlags(
        production=production,
        challenger=challenger,
        reliability_weight_mode=mode,
        stages=stages,
    )


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in _TRUE


def apply_production_flags_to_config(cfg: object, resolved: ResolvedFlags) -> None:
    """Mutate an MLConfig-like object so production path matches resolved.production.

    Only sets attributes that exist on *cfg*. Does not enable challenger path.
    """
    for name, value in resolved.production.items():
        if hasattr(cfg, name):
            setattr(cfg, name, bool(value))
    if hasattr(cfg, "reliability_weight_mode"):
        setattr(cfg, "reliability_weight_mode", resolved.reliability_weight_mode)


def apply_challenger_flags_to_config(cfg: object, resolved: ResolvedFlags) -> None:
    """Mutate config to the challenger (shadow) path for dual scoring."""
    for name, value in resolved.challenger.items():
        if hasattr(cfg, name):
            setattr(cfg, name, bool(value))
    # Challenger may evaluate continuous mode even while production stays bucket
    if hasattr(cfg, "reliability_weight_mode") and resolved.any_challenger():
        if any(resolved.challenger.values()):
            # Prefer continuous when observing continuous mode in shadow
            setattr(cfg, "reliability_weight_mode", "continuous")
