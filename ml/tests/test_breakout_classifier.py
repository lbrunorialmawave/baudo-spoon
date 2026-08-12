"""Unit tests for the breakout classifier (PR7)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.breakout import (
    BreakoutClassifier,
    BreakoutClassifierMetrics,
    evaluate_breakout_classifier,
    train_breakout_classifier,
)


def _toy_dataset(n_pos: int = 30, n_neg: int = 70) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small, separable breakout dataset for sanity tests."""
    rng = np.random.default_rng(0)
    pos = pd.DataFrame(
        {
            "mins_played_lag1": rng.normal(200, 30, n_pos),
            "starts_lag1": rng.normal(8, 3, n_pos),
            "appearances_lag1": rng.normal(20, 5, n_pos),
        }
    )
    neg = pd.DataFrame(
        {
            "mins_played_lag1": rng.normal(150, 20, n_neg),
            "starts_lag1": rng.normal(2, 1, n_neg),
            "appearances_lag1": rng.normal(10, 3, n_neg),
        }
    )
    X = pd.concat([pos, neg], ignore_index=True)
    y = pd.Series([1] * n_pos + [0] * n_neg, name="breakout")
    return X, y


class TestBreakoutClassifier:
    def test_fit_and_predict_shape(self) -> None:
        X, y = _toy_dataset()
        clf = train_breakout_classifier(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X),)
        assert ((0.0 <= proba) & (proba <= 1.0)).all()

    def test_predict_threshold(self) -> None:
        X, y = _toy_dataset()
        clf = BreakoutClassifier(positive_threshold=0.7).fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        # Each predicted positive must have proba >= 0.7
        for p, d in zip(proba, preds):
            if d == 1:
                assert p >= 0.7 - 1e-9

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            BreakoutClassifier(positive_threshold=0.0)
        with pytest.raises(ValueError):
            BreakoutClassifier(positive_threshold=1.0)

    def test_unfitted_classifier_raises(self) -> None:
        clf = BreakoutClassifier()
        with pytest.raises(RuntimeError):
            clf.predict_proba(pd.DataFrame({"a": [1.0]}))

    def test_empty_training_set_raises(self) -> None:
        clf = BreakoutClassifier()
        with pytest.raises(ValueError):
            clf.fit(pd.DataFrame(), pd.Series([], dtype=int))

    def test_separable_dataset_achieves_high_auc(self) -> None:
        X, y = _toy_dataset()
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y,
        )
        clf = train_breakout_classifier(X_train, y_train)
        metrics = evaluate_breakout_classifier(clf, X_test, y_test)
        assert isinstance(metrics, BreakoutClassifierMetrics)
        # On a clearly separable dataset the AUC should be well above 0.7.
        assert metrics.roc_auc > 0.7
        assert metrics.n_test == len(X_test)
        # brier in [0, 1]
        assert 0.0 <= metrics.brier_score <= 1.0

    def test_metrics_summary_cached(self) -> None:
        X, y = _toy_dataset()
        clf = train_breakout_classifier(X, y)
        m1 = evaluate_breakout_classifier(clf, X, y)
        m2 = clf.score(X, y)
        # Cached on the instance, so re-eval returns equal numbers
        assert m1.roc_auc == pytest.approx(m2.roc_auc)
