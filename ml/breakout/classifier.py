"""Lightweight breakout classifier (PR7 of the low-sample plan).

The breakout dataset produced by :mod:`ml.breakout.dataset` is **highly
imbalanced** (base rate typically < 5%).  Training a production-grade
classifier therefore requires a *deliberate* modelling choice: in
this module we ship a small, conservative baseline — a logistic
regression on standardised, lagged features — and leave a single
extension point (``spec.estimator``) so that future PRs can swap in a
gradient-boosted model without touching the training pipeline.

The classifier is intentionally small (≤ 200 LOC) because the **value
of this PR is the dataset and the leakage guarantees**, not the model
itself.  The plan is explicit:

> "Il PR7 (breakout model) è inizialmente un semplice flag/baseline
>  che non viene usato in produzione finché le metriche offline non
>  sono soddisfacenti."

Public surface:

* :class:`BreakoutClassifier` — trains and predicts breakout probability.
* :func:`train_breakout_classifier` — high-level helper.
* :func:`evaluate_breakout_classifier` — offline metrics.

The class never reads or writes the global config; it accepts the
``MLConfig`` instance and reads **only** the breakout-related fields
if it needs to (currently none).  This keeps it composable with the
offline experiment harness (PR5) and easy to mock in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

DEFAULT_BREAKOUT_RANDOM_SEED: Final[int] = 42


# ── Public DTO ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BreakoutClassifierMetrics:
    """Offline metrics for the breakout classifier.

    All metrics are computed on the *test* split of the breakout
    dataset.  The instance is JSON-serialisable; embed it in the
    trainer output and in the experiment report.
    """

    n_train: int
    n_test: int
    base_rate_test: float
    roc_auc: float
    average_precision: float
    brier_score: float
    positive_threshold: float


# ── Classifier ─────────────────────────────────────────────────────────────

class BreakoutClassifier:
    """Train / predict wrapper for the breakout classification task.

    Args:
        random_seed: Seed for the underlying classifier.  Defaults to
            :data:`DEFAULT_BREAKOUT_RANDOM_SEED`.
        positive_threshold: Probability threshold for the binary
            decision (also reported as part of the metrics).
    """

    def __init__(
        self,
        random_seed: int = DEFAULT_BREAKOUT_RANDOM_SEED,
        positive_threshold: float = 0.5,
    ) -> None:
        if not 0.0 < positive_threshold < 1.0:
            raise ValueError(
                f"positive_threshold must be in (0, 1), got {positive_threshold}"
            )
        self.random_seed = random_seed
        self.positive_threshold = positive_threshold
        self._pipeline: Pipeline | None = None
        self._metrics: BreakoutClassifierMetrics | None = None

    @property
    def metrics(self) -> BreakoutClassifierMetrics | None:
        return self._metrics

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BreakoutClassifier":
        """Fit the classifier on the (features, labels) pair.

        Pre-processing is a simple ``SimpleImputer`` followed by a
        ``StandardScaler``; the head is an L2-regularised logistic
        regression.  This choice is deliberately conservative: the
        dataset is small and imbalanced, and a non-linear model would
        overfit.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty training set")
        self._pipeline = _build_pipeline(self.random_seed)
        log.info("Fitting BreakoutClassifier on %d rows …", len(X))
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probability for each row."""
        if self._pipeline is None:
            raise RuntimeError("Classifier not fitted; call .fit() first")
        return self._pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the binary decision (0/1) at ``positive_threshold``."""
        proba = self.predict_proba(X)
        return (proba >= self.positive_threshold).astype(int)

    def score(self, X: pd.DataFrame, y: pd.Series) -> BreakoutClassifierMetrics:
        """Compute offline metrics and cache them on the instance."""
        if self._pipeline is None:
            raise RuntimeError("Classifier not fitted; call .fit() first")
        proba = self.predict_proba(X)
        # Edge case: only one class in the test set.
        unique = np.unique(y)
        if unique.size < 2:
            log.warning(
                "Test set has only one class (n=%d); reporting NaN for ranking metrics.",
                unique.size,
            )
            roc_auc = float("nan")
            ap = float("nan")
        else:
            roc_auc = float(roc_auc_score(y, proba))
            ap = float(average_precision_score(y, proba))
        brier = float(brier_score_loss(y, proba))
        self._metrics = BreakoutClassifierMetrics(
            n_train=len(self._pipeline.steps[-1][1].classes_),  # not great
            n_test=len(X),
            base_rate_test=float(y.mean()),
            roc_auc=roc_auc,
            average_precision=ap,
            brier_score=brier,
            positive_threshold=float(self.positive_threshold),
        )
        return self._metrics


# ── High-level helpers ──────────────────────────────────────────────────────

def train_breakout_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_seed: int = DEFAULT_BREAKOUT_RANDOM_SEED,
) -> BreakoutClassifier:
    """Fit a default :class:`BreakoutClassifier` and return it."""
    return BreakoutClassifier(random_seed=random_seed).fit(X, y)


def evaluate_breakout_classifier(
    classifier: BreakoutClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> BreakoutClassifierMetrics:
    """Score a fitted classifier on a test set."""
    return classifier.score(X, y)


# ── Internal helpers ────────────────────────────────────────────────────────

def _build_pipeline(random_seed: int) -> Pipeline:
    """Return the default feature-processing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
