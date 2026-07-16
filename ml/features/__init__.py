"""Feature engineering classes for the player evaluation system.

These classes wrap the logic from ``ml/preprocessing/features.py`` as
stateless, independently-testable ``Feature`` subclasses.  The original
preprocessing module remains operational during Phase 1; these classes are
used by Phase 3+ ensemble training.
"""
from ml.features.per90 import ALL_PER90_FEATURES
from ml.features.rolling import ALL_DELTA_FEATURES, ALL_ROLLING_FEATURES, ALL_TREND_FEATURES
from ml.features.sap import ALL_SAP_FEATURES
from ml.features.team_strength import IsTopTeamFeature, TeamRankNormFeature, TeamStrengthFeature
from ml.features.base import REGISTRY, FeatureRegistry, compute_feature_matrix

__all__ = [
    "ALL_PER90_FEATURES",
    "ALL_TREND_FEATURES",
    "ALL_ROLLING_FEATURES",
    "ALL_DELTA_FEATURES",
    "ALL_SAP_FEATURES",
    "TeamStrengthFeature",
    "IsTopTeamFeature",
    "TeamRankNormFeature",
    "REGISTRY",
    "FeatureRegistry",
    "compute_feature_matrix",
]
