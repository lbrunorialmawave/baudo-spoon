from __future__ import annotations

"""sklearn preprocessing pipeline construction.

Produces a ``ColumnTransformer`` that:
- Imputes missing numeric values with the column median.
- Scales numeric features with ``RobustScaler`` (IQR-based; resistant to the
  long-tailed distributions produced by per-90 stat normalisation).
- One-hot encodes categorical features (unknown categories → all zeros).

Design notes:
- ``SimpleImputer`` uses median strategy, which is robust to outliers.
- ``RobustScaler`` is applied after imputation so it never sees NaN.
  Preferred over ``StandardScaler`` because per-90 stats (goals, shots, …)
  exhibit heavy right tails; the IQR-based scale is unaffected by extreme
  values, producing more stable feature magnitudes for gradient-based models.
- ``OneHotEncoder(handle_unknown='ignore')`` silently ignores categories
  not seen during training — important at inference time.
- The pipeline is stateless until fitted; fitting is done inside
  ``pipeline.trainer.Trainer``.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Return an unfitted ``ColumnTransformer`` preprocessor.

    Numeric features are median-imputed then scaled with ``RobustScaler``
    (centre = median, scale = IQR).  This is preferable to ``StandardScaler``
    for per-90 football statistics, whose distributions are right-skewed with
    occasional extreme outliers (e.g. a striker who scored 1.5 goals/90).

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        A ``ColumnTransformer`` with two sub-pipelines:
        ``numeric`` (impute → RobustScaler) and ``categorical`` (OHE).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple] = [
        ("numeric", numeric_pipeline, numeric_features),
    ]
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the ordered feature names after the preprocessor is fitted.

    This is a convenience wrapper around ``get_feature_names_out`` so that
    SHAP plots and feature importance charts are labelled correctly.
    """
    return list(preprocessor.get_feature_names_out())
