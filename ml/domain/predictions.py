from __future__ import annotations

from dataclasses import dataclass

SHAP_TOLERANCE: float = 1e-4
"""Maximum allowed deviation: |Σ shap_values + base_value - prediction|."""


@dataclass(frozen=True)
class PredictionExplanation:
    """Full explanation for a single player prediction.

    Args:
        prediction: The model's point prediction.
        confidence: Calibrated confidence score in [0.0, 1.0].
        variance: Estimated predictive variance (>= 0).
        prediction_interval: (lower, upper) credible interval.
        best_case: Optimistic scenario value.
        worst_case: Pessimistic scenario value.
        top_features: List of (feature_name, shap_value) sorted by |shap_value| desc.
        shap_values: Full dict of feature -> shap contribution.
        base_value: SHAP base value (expected model output over training set).
    """

    prediction: float
    confidence: float
    variance: float
    prediction_interval: tuple[float, float]
    best_case: float
    worst_case: float
    top_features: list[tuple[str, float]]
    shap_values: dict[str, float]
    base_value: float

    def __post_init__(self) -> None:
        lo, hi = self.prediction_interval
        if lo > hi:
            raise ValueError(
                f"prediction_interval lower bound {lo} > upper bound {hi}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.variance < 0.0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")

    def shap_coherence_error(self) -> float:
        """Return |Σ shap_values + base_value - prediction|.

        Should be < SHAP_TOLERANCE for a well-formed explanation.
        """
        return abs(sum(self.shap_values.values()) + self.base_value - self.prediction)

    def is_shap_coherent(self) -> bool:
        """True when SHAP values are consistent with the prediction."""
        return self.shap_coherence_error() < SHAP_TOLERANCE
