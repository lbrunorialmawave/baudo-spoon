"""Public surface for the breakout-dataset module (PR6) and the
breakout-classifier module (PR7).
"""

from .classifier import (
    BreakoutClassifier,
    BreakoutClassifierMetrics,
    evaluate_breakout_classifier,
    train_breakout_classifier,
)
from .dataset import (
    DEFAULT_BREAKOUT_TARGET_MINUTES,
    DEFAULT_FEATURE_LAG_SEASONS,
    BreakoutDatasetStats,
    build_breakout_dataset,
    build_breakout_labels,
    engineer_breakout_features,
)

__all__ = [
    # PR6
    "DEFAULT_BREAKOUT_TARGET_MINUTES",
    "DEFAULT_FEATURE_LAG_SEASONS",
    "BreakoutDatasetStats",
    "build_breakout_dataset",
    "build_breakout_labels",
    "engineer_breakout_features",
    # PR7
    "BreakoutClassifier",
    "BreakoutClassifierMetrics",
    "evaluate_breakout_classifier",
    "train_breakout_classifier",
]
