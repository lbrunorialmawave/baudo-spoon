from __future__ import annotations

"""sklearn preprocessing pipeline construction.

Produces a ``ColumnTransformer`` that:
- Imputes **event-based** numeric features (per-90 stats: goals, assists, …)
  with **zero** — a missing event means the event did not occur.
- Imputes **environmental/contextual** numeric features (team strength,
  quotation signals, …) with the **column median** — a missing context
  means "unknown context", not "zero context".
- Scales all numeric features with ``RobustScaler`` (IQR-based; resistant to
  the long-tailed distributions produced by per-90 stat normalisation).
- One-hot encodes categorical features (unknown categories → all zeros).

Design notes:
- Differentiated imputation (zero for events, median for context) prevents
  the phantom-zero-context bias that would arise from median-imputing goals.
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
    environmental_features: list[str] | None = None,
) -> ColumnTransformer:
    """Return an unfitted ``ColumnTransformer`` preprocessor.

    Differentiated imputation strategy:
      * **Event-based** features (the default — any numeric feature not listed
        in *environmental_features*) are imputed with **0**.  A missing goal
        count means the player did not score.
      * **Environmental/contextual** features (e.g. team strength, quotation
        signals) are imputed with the **column median**.  A missing context
        means "unknown", not "zero".
      * All numeric features are then scaled with ``RobustScaler``
        (centre = median, scale = IQR).

    Args:
        numeric_features: List of all numeric column names (both event-based
            and environmental).  Must be a superset of *environmental_features*.
        categorical_features: List of categorical column names.
        environmental_features: Subset of *numeric_features* that represent
            contextual/environmental signals and should be median-imputed.
            If ``None`` or empty, all numeric features are zero-imputed
            (legacy behaviour).

    Returns:
        A ``ColumnTransformer`` with sub-pipelines:
        ``numeric_event`` (impute=0 → RobustScaler),
        ``numeric_env`` (impute=median → RobustScaler),
        ``categorical`` (OHE).
    """
    # Split numeric features into event-based (default) and environmental
    if environmental_features:
        event_features = [
            f for f in numeric_features if f not in environmental_features
        ]
    else:
        event_features = list(numeric_features)

    transformers: list[tuple] = []

    # ── Event-based features: missing = 0 ─────────────────────────────────
    if event_features:
        event_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("numeric_event", event_pipeline, event_features))

    # ── Environmental features: missing = median ─────────────────────────
    env_list = environmental_features or []
    if env_list:
        env_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("numeric_env", env_pipeline, env_list))

    # ── Categorical features ─────────────────────────────────────────────
    if categorical_features:
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
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the ordered feature names after the preprocessor is fitted.

    This is a convenience wrapper around ``get_feature_names_out`` so that
    SHAP plots and feature importance charts are labelled correctly.
    """
    return list(preprocessor.get_feature_names_out())
