"""Feature registry and batch computation utilities."""

from __future__ import annotations

import logging

import polars as pl

from ml.domain.features import Feature

__all__ = ["REGISTRY", "FeatureRegistry", "compute_feature_matrix"]

log = logging.getLogger(__name__)


class FeatureRegistry:
    """Holds named Feature instances; populated at import time by feature modules."""

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def register(self, feature: Feature) -> Feature:
        """Register a feature; raises ValueError on duplicate name."""
        if feature.name in self._features:
            raise ValueError(f"Feature '{feature.name}' already registered.")
        self._features[feature.name] = feature
        return feature

    def get(self, name: str) -> Feature:
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._features[name]

    def all(self) -> list[Feature]:
        return list(self._features.values())

    def names(self) -> list[str]:
        return list(self._features.keys())

    def for_role(self, role: str) -> list[Feature]:
        """Return features applicable to *role* ('GK', 'DEF', 'MID', 'FWD').

        Features without an ``applicable_roles`` attribute are included for all roles.
        """
        return [
            f
            for f in self._features.values()
            if not hasattr(f, "applicable_roles") or role in f.applicable_roles
        ]


# Module-level singleton registry.
REGISTRY = FeatureRegistry()


def compute_feature_matrix(
    data: pl.DataFrame,
    features: list[Feature],
    *,
    on_error: str = "raise",  # "raise" | "skip"
) -> pl.DataFrame:
    """Apply a list of features to *data*, returning one column per feature.

    Args:
        data: Input Polars DataFrame.
        features: Feature instances to compute.
        on_error: ``"raise"`` re-raises computation errors; ``"skip"`` logs
            and omits the failed feature column.

    Returns:
        DataFrame with len(data) rows and one column per successfully computed
        feature. Empty DataFrame when all features fail or *features* is empty.
    """
    computed: dict[str, pl.Series] = {}
    for feat in features:
        try:
            s = feat.safe_compute(data)
            computed[feat.name] = s.rename(feat.name)
        except Exception as exc:
            if on_error == "raise":
                raise
            log.warning("Feature '%s' failed: %s — skipping.", feat.name, exc)

    if not computed:
        return pl.DataFrame()
    return pl.DataFrame(computed)
