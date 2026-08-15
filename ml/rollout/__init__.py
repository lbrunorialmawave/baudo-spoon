"""Public surface for the production-rollout module (PR8)."""

from .observability import CohortObservability, compute_cohort_observability, diagnostic_score_layers
from .env_flags import ResolvedFlags, apply_challenger_flags_to_config, apply_production_flags_to_config, resolve_env_flags
from .controller import (
    DEFAULT_ROLLOUT_PCT,
    FeatureFlag,
    FlagStage,
    RolloutController,
    ShadowComparison,
    default_controllers,
    reliability_weight_mode_for_stage,
    shadow_compare,
)

__all__ = [
    "ResolvedFlags",
    "resolve_env_flags",
    "apply_production_flags_to_config",
    "apply_challenger_flags_to_config",
    "CohortObservability",
    "compute_cohort_observability",
    "diagnostic_score_layers",
    "FeatureFlag",
    "FlagStage",
    "RolloutController",
    "ShadowComparison",
    "shadow_compare",
    "default_controllers",
    "DEFAULT_ROLLOUT_PCT",
    "reliability_weight_mode_for_stage",
]
