"""Public surface for the offline experiment harness (PR5)."""

from .harness import (
    DEFAULT_VARIANTS,
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    ExperimentVariant,
    apply_variant,
    default_variants,
    run_experiment,
)

__all__ = [
    "DEFAULT_VARIANTS",
    "VARIANT_A",
    "VARIANT_B",
    "VARIANT_C",
    "VARIANT_D",
    "ExperimentVariant",
    "apply_variant",
    "default_variants",
    "run_experiment",
]
