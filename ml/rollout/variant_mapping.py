"""Canonical FeatureFlag → harness-variant mapping (WS14-bis).

Single source of truth shared by:
* ``ml.scripts.check_promotion_gate`` (which variant to verify)
* ``ml.run_pipeline._select_variant_label`` (which variant to label)
* ``ml.experiments.harness`` (canonical definition of A/B/C/D)

If a fifth flag is added in the future, this is the only file that
needs to be touched so that gate, run_pipeline and harness stay
consistent.
"""
from __future__ import annotations

from typing import Final

from ml.rollout.controller import FeatureFlag

# Precedence order: when multiple boolean flags are ACTIVE at the same
# time, the first entry in the dict wins (mirrors the order already
# used in run_pipeline._select_variant_label).
FLAG_TO_VARIANT: Final[dict[str, str]] = {
    FeatureFlag.PER90_SHRINKAGE.value: "C_shrinkage",
    FeatureFlag.LIMITED_SAMPLE_TRAINING.value: "B_weighting",
    FeatureFlag.RECENT_ROLE_FEATURES.value: "D_recent_role_features",
}
DEFAULT_VARIANT: Final[str] = "A_control"

# Alias used by provenance checks and gate defaults.
DEFAULT_A_CONTROL_LABEL: Final[str] = DEFAULT_VARIANT


def variant_for_flag(flag_value: str) -> str:
    """Return the harness label expected for the given flag.

    Raises ``KeyError`` for a flag that has no corresponding harness
    variant (e.g. ``enable_breakout_model``, ``reliability_weight_mode``
    — these are outside the A/B/C/D matrix and must not go through this
    gate).
    """
    if flag_value not in FLAG_TO_VARIANT:
        raise KeyError(
            f"flag {flag_value!r} has no corresponding harness variant "
            "— check_promotion_gate is not applicable; use a different "
            "promotion path for this flag."
        )
    return FLAG_TO_VARIANT[flag_value]
