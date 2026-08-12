"""Public surface for the production-rollout module (PR8)."""

from .controller import (
    DEFAULT_ROLLOUT_PCT,
    FeatureFlag,
    FlagStage,
    RolloutController,
    ShadowComparison,
    default_controllers,
    shadow_compare,
)

__all__ = [
    "FeatureFlag",
    "FlagStage",
    "RolloutController",
    "ShadowComparison",
    "shadow_compare",
    "default_controllers",
    "DEFAULT_ROLLOUT_PCT",
]
