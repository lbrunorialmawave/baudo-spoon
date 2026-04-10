from __future__ import annotations

"""Model explainability: SHAP values and feature importance.

Two explainability strategies:
1. **Tree-based SHAP** (``shap.TreeExplainer``) — exact Shapley values for
   tree ensembles (Random Forest, GBM, XGBoost).  Fast and accurate.
2. **Permutation importance** — model-agnostic fallback used for Ridge and
   any other linear/non-tree model.  Also useful as a sanity check against
   SHAP for tree models.

Both produce:
- A ranked feature importance DataFrame.
- A SHAP summary plot (bar + beeswarm) saved to the artifacts directory.

Design notes:
- The SHAP explainer is run on a subsample (``cfg.shap_sample_size``) to
  keep computation tractable for large datasets.
- ``shap`` is an optional import: if not installed, explainability falls
  back to permutation importance only.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
log = logging.getLogger(__name__)

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    log.warning("shap not installed; SHAP values will be skipped.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_preprocessor(pipeline: Pipeline):
    """Return the preprocessor step from a Pipeline."""
    return pipeline.named_steps.get("preprocessor")


def _get_model(pipeline: Pipeline):
    """Return the model step from a Pipeline."""
    return pipeline.named_steps.get("model")


def _is_tree_model(model) -> bool:
    """Check if the model is a tree-based estimator."""
    tree_types = (
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
    )
    return type(model).__name__ in tree_types


# ── Permutation importance ────────────────────────────────────────────────────

def compute_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of permutation importances, sorted descending."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    # feature_names here are the *raw* column names (pre-transformation)
    imp_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp_df


# ── Tree feature importance ───────────────────────────────────────────────────

def compute_tree_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract built-in feature importances from a tree-based model."""
    model = _get_model(pipeline)
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} does not expose feature_importances_.")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap_values(
    pipeline: Pipeline,
    X_raw: pd.DataFrame,
    feature_names: list[str],
    sample_size: int = 300,
    random_seed: int = 42,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Compute SHAP values for the model.

    Args:
        pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'model' steps.
        X_raw: Raw (un-transformed) feature DataFrame.
        feature_names: Feature names after preprocessing (for plot labels).
        sample_size: Number of rows to subsample for SHAP computation.
        random_seed: RNG seed for subsampling.

    Returns:
        (shap_values, X_transformed_sample) or None if SHAP is unavailable.
    """
    if not _HAS_SHAP:
        return None

    preprocessor = _get_preprocessor(pipeline)
    model = _get_model(pipeline)

    if preprocessor is None or model is None:
        log.warning("Pipeline does not have 'preprocessor' and 'model' steps.")
        return None

    rng = np.random.default_rng(random_seed)
    n = min(sample_size, len(X_raw))
    idx = rng.choice(len(X_raw), size=n, replace=False)
    X_sample_raw = X_raw.iloc[idx]
    X_sample = preprocessor.transform(X_sample_raw)

    try:
        if _is_tree_model(model):
            explainer = _shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
        else:
            explainer = _shap.LinearExplainer(
                model,
                _shap.maskers.Independent(X_sample),
            )
            shap_vals = explainer.shap_values(X_sample)
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
        return None

    return shap_vals, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    output_path: str,
    model_name: str = "",
) -> None:
    """Save a SHAP beeswarm + bar summary plot."""
    if not _HAS_SHAP:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plt.sca(axes[0])
    _shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=20,
    )
    axes[0].set_title(f"SHAP Beeswarm — {model_name}")

    plt.sca(axes[1])
    _shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    axes[1].set_title(f"SHAP Feature Importance — {model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP summary plot saved to %s", output_path)


def plot_feature_importance(
    imp_df: pd.DataFrame,
    output_path: str,
    model_name: str = "",
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of top-N feature importances."""
    df = imp_df.head(top_n).iloc[::-1]  # reverse for bottom-to-top display

    imp_col = "importance" if "importance" in df.columns else "importance_mean"
    err_col = "importance_std" if "importance_std" in df.columns else None

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    xerr = df[err_col].values if err_col else None
    ax.barh(
        df["feature"], df[imp_col],
        xerr=xerr, color="steelblue", alpha=0.8,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Feature importance plot saved to %s", output_path)
