from __future__ import annotations

"""Abstract base class and registry for regression models.

Design:
- Every model wraps a scikit-learn estimator (or an XGBoost estimator that
  implements the sklearn API).
- ``ModelSpec`` is a thin data class that bundles: name, estimator instance,
  and an optional hyperparameter search space used during tuning.
- ``MODEL_REGISTRY`` is a dict of all available models, keyed by name.
  The Trainer selects from here.
"""

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import RegressorMixin


@dataclass
class ModelSpec:
    """Descriptor for a regression model.

    Attributes:
        name: Human-readable identifier (also used as the artifact file stem).
        estimator: An unfitted sklearn-compatible regressor instance.
        param_grid: Hyperparameter search space for ``RandomizedSearchCV``.
            Each key must be a valid ``estimator`` init-param name.
            Set to ``{}`` for models that should not be tuned.
    """

    name: str
    estimator: RegressorMixin
    param_grid: dict[str, Any] = field(default_factory=dict)
