from __future__ import annotations

"""Player clustering for low-cost alternative discovery.

Approach:
1. Select per-90 normalised production features (goals, assists, shots, …).
2. Impute + scale with StandardScaler.
3. Reduce with PCA (retain ``pca_variance_threshold`` of variance).
4. Run KMedoids clustering (``n_clusters`` from config).
   When ``n_clusters == -1``, the optimal K is selected automatically via
   the Silhouette method (maximise mean silhouette score over k ∈ [2, 10]).
5. For every top player (defined as predicted fantavoto ≥ ``top_percentile``),
   find players in the **same cluster** whose FotMob team rank (proxy for
   market value) is significantly lower — these are "low-cost copies".

Similarity metric:
- Euclidean distance in PCA-reduced space (lower = more similar profile).

Visualisation:
- A combined figure with two panels:
  (a) 2-D PCA scatter coloured by cluster; top 10% annotated with names.
  (b) Silhouette plot showing per-cluster coefficient distributions.
  Saved to ``artifacts/cluster_viz.png``.

Design notes:
- KMedoids (scikit-learn-extra) is preferred over KMeans for reduced outlier
  sensitivity; falls back to KMeans if scikit-learn-extra is not installed.
- All randomness is seeded via ``cfg.random_seed``.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import MLConfig

matplotlib.use("Agg")  # non-interactive backend for server environments
log = logging.getLogger(__name__)

# ── Optional KMedoids import ──────────────────────────────────────────────────

try:
    from sklearn_extra.cluster import KMedoids  # type: ignore[import]
    _HAS_KMEDOIDS = True
except ImportError:
    _HAS_KMEDOIDS = False
    log.warning(
        "scikit-learn-extra not installed; falling back to KMeans. "
        "Install with: pip install scikit-learn-extra"
    )

# ── Feature selection for clustering ─────────────────────────────────────────
# We cluster on *style* (per-90 production); we intentionally EXCLUDE
# team-context and temporal features so that players from different teams
# and eras can be compared on pure performance profile.
_CLUSTER_FEATURE_CANDIDATES = [
    "goals_per90",
    "goal_assist_per90",
    "total_scoring_att_per90",
    "ontarget_scoring_att_per90",
    "won_contest_per90",
    "total_att_assist_per90",
    "big_chance_created_per90",
    "yellow_card_per90",
    "interception_per90",
    "total_tackle_per90",
    "effective_clearance_per90",
    "fouls_per90",
]

# ── Role-constraint constants ─────────────────────────────────────────────────
# Maps canonical_role strings to integer codes used in the distance penalty.
_ROLE_TO_INT: dict[str, int] = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# Additive penalty (in PCA Euclidean units) applied to cross-role pairwise
# distances when building the role-aware KMedoids distance matrix.
# Set large enough to strongly discourage cross-role cluster formation,
# but not so extreme as to force pure within-role clusters when data is sparse.
_CROSS_ROLE_PENALTY: float = 5.0


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    """Output of :func:`run_clustering`."""

    df: pd.DataFrame
    """Player-season DataFrame with ``cluster_id`` and ``pca_*`` columns."""

    cluster_labels: np.ndarray
    """Cluster label array aligned with ``df``."""

    pca_components: pd.DataFrame
    """PCA-reduced coordinates (shape: n_rows × n_pca_components)."""

    inertia: float
    """Within-cluster sum of squares (KMeans) or sum of distances (KMedoids)."""

    silhouette: float
    """Mean Silhouette score of the clustering."""

    explained_variance: list[float]
    """Fraction of variance explained by each PCA component."""

    feature_names: list[str]
    """Clustering feature names (after PCA: 'pca_0', 'pca_1', …)."""

    n_clusters_used: int = field(default=0)
    """Actual number of clusters used (resolved after auto-K selection)."""


@dataclass
class LowCostAlternative:
    """One low-cost alternative to a top player."""

    top_player_id: int
    top_player_name: str
    top_player_team: str
    top_player_fantavoto: float

    alt_player_id: int
    alt_player_name: str
    alt_player_team: str
    alt_player_fantavoto: float

    cluster_id: int
    distance: float  # Euclidean in PCA space


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_cluster_features(df: pd.DataFrame) -> list[str]:
    return [c for c in _CLUSTER_FEATURE_CANDIDATES if c in df.columns and df[c].notna().any()]


def _make_clusterer(n_clusters: int, random_seed: int):
    """Return a KMedoids (or KMeans fallback) instance.

    KMeans uses ``n_init=20`` to guarantee convergence stability against
    high-dimensional local minima (default ``"auto"`` is insufficient for
    the PCA-reduced feature spaces encountered in this pipeline).
    """
    if _HAS_KMEDOIDS:
        return KMedoids(n_clusters=n_clusters, random_state=random_seed, metric="euclidean")
    return KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)


def _select_k_by_silhouette(
    X: np.ndarray,
    k_range: range,
    random_seed: int,
) -> int:
    """Select optimal K by maximising mean Silhouette score.

    Iterates over *k_range*, fits a clusterer, and returns the K that
    yields the highest mean silhouette coefficient.

    Args:
        X: Pre-processed feature matrix.
        k_range: Range of K values to evaluate.
        random_seed: Reproducibility seed.

    Returns:
        The K with the highest silhouette score in *k_range*.
    """
    best_k = k_range.start
    best_score = -1.0

    for k in k_range:
        if k >= len(X):
            break
        clusterer = _make_clusterer(k, random_seed)
        labels = clusterer.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        log.debug("Silhouette k=%d → %.4f", k, score)
        if score > best_score:
            best_score, best_k = score, k

    log.info(
        "SilhouetteScorer selected K=%d (mean silhouette=%.4f)", best_k, best_score
    )
    return best_k


def _elbow_suggestor(
    X: np.ndarray,
    k_range: range,
    random_seed: int,
) -> int:
    """Return the elbow point via the second-difference method (legacy)."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
        km.fit(X)
        inertias.append(km.inertia_)

    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    suggested = k_range[int(np.argmin(second_diffs)) + 2]
    log.info(
        "Elbow method suggests K=%d (range %d–%d).",
        suggested, k_range.start, k_range.stop - 1,
    )
    return suggested


def _build_role_aware_distance_matrix(
    X_pca: np.ndarray,
    role_codes: np.ndarray,
    cross_role_penalty: float = _CROSS_ROLE_PENALTY,
) -> np.ndarray:
    """Build a pairwise Euclidean distance matrix with cross-role penalties.

    Intra-role pairs retain their pure Euclidean distance.  Cross-role pairs
    (e.g. GK vs DEF) receive an additive penalty of *cross_role_penalty*,
    ensuring KMedoids strongly prefers same-role cluster formation.

    This implements "soft" role constraints: a cross-role assignment is still
    possible when the within-cluster inertia benefit outweighs the penalty,
    but it will be significantly discouraged.

    Complexity: O(N² · D) where D is the PCA dimensionality.  Acceptable for
    typical season sizes (N < 1000).

    Args:
        X_pca: PCA-reduced feature matrix (shape: N × D).
        role_codes: Integer role code per player (shape: N,), from _ROLE_TO_INT.
        cross_role_penalty: Additive distance penalty for cross-role pairs.

    Returns:
        Symmetric distance matrix of shape (N, N).
    """
    # Vectorised pairwise Euclidean distance — O(N² · D)
    diff = X_pca[:, np.newaxis, :] - X_pca[np.newaxis, :, :]  # (N, N, D)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))                # (N, N)

    # Boolean cross-role mask: True where roles differ
    role_mismatch = role_codes[:, np.newaxis] != role_codes[np.newaxis, :]  # (N, N)
    dists = dists + role_mismatch.astype(np.float64) * cross_role_penalty

    # Ensure strict symmetry and zero diagonal (numerical safety)
    np.fill_diagonal(dists, 0.0)
    return dists

def run_clustering(
    df: pd.DataFrame,
    cfg: MLConfig,
) -> ClusterResult:
    """Fit PCA + KMedoids on *df* and return a :class:`ClusterResult`.

    When ``cfg.n_clusters == -1``, the optimal K is selected automatically
    via :func:`_select_k_by_silhouette` over k ∈ [2, min(10, n_rows)].

    **Soft-role constraints**: If *df* contains a ``canonical_role`` column,
    pairwise distances are augmented with :data:`_CROSS_ROLE_PENALTY` for
    cross-role player pairs.  This discourages KMedoids from grouping a GK
    with a DEF (or any other cross-role combination) while still allowing it
    when the performance profiles are sufficiently similar.  The KMeans
    fallback instead appends a weighted role coordinate to the PCA matrix.

    Args:
        df: Feature-engineered player-season DataFrame (after
            :func:`~preprocessing.features.engineer_features`).
            Must contain ``fantavoto_medio`` or ``predicted_fantavoto``.
        cfg: Pipeline configuration.

    Returns:
        A :class:`ClusterResult` with cluster assignments and PCA coordinates.
    """
    feat_cols = _select_cluster_features(df)
    if not feat_cols:
        raise ValueError(
            "No clustering features found. "
            "Ensure per-90 features were computed before clustering."
        )
    log.info("Clustering on %d features: %s", len(feat_cols), feat_cols)

    # ── Impute + scale ────────────────────────────────────────────────────────
    X_raw = df[feat_cols].values.astype(float)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_full = PCA(random_state=cfg.random_seed)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, cfg.pca_variance_threshold)) + 1
    n_components = max(2, min(n_components, len(feat_cols)))  # at least 2 for viz

    log.info(
        "PCA: retaining %d components (%.1f%% variance).",
        n_components, cumvar[n_components - 1] * 100,
    )

    pca = PCA(n_components=n_components, random_state=cfg.random_seed)
    X_pca = pca.fit_transform(X_scaled)

    # ── K selection (on pure Euclidean PCA space) ─────────────────────────────
    n_clusters = cfg.n_clusters
    if n_clusters == -1:
        log.info("n_clusters=-1: auto-selecting K via SilhouetteScorer …")
        k_range = range(2, min(11, len(df)))
        n_clusters = _select_k_by_silhouette(X_pca, k_range, cfg.random_seed)

    # ── Extract role codes for soft-role constraint ────────────────────────────
    role_codes: Optional[np.ndarray] = None
    if "canonical_role" in df.columns:
        role_codes = (
            df["canonical_role"]
            .map(_ROLE_TO_INT)
            .fillna(2)          # unknown roles default to MID (2)
            .values.astype(np.int32)
        )
        log.info(
            "Soft role constraints enabled (cross-role penalty=%.1f).",
            _CROSS_ROLE_PENALTY,
        )

    # ── KMedoids with role-aware distance OR KMeans fallback ─────────────────
    if role_codes is not None and _HAS_KMEDOIDS:
        dist_matrix = _build_role_aware_distance_matrix(X_pca, role_codes)
        clusterer = KMedoids(
            n_clusters=n_clusters,
            random_state=cfg.random_seed,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(dist_matrix)
        inertia = float(getattr(clusterer, "inertia_", float("nan")))
        algo = "KMedoids+RoleConstraint"
    elif role_codes is not None:
        # KMeans fallback: append scaled role coordinate to nudge same-role
        # players together without hard enforcement.
        role_col = (role_codes / 3.0 * _CROSS_ROLE_PENALTY)[:, np.newaxis]
        X_role_aug = np.hstack([X_pca, role_col])
        clusterer = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed, n_init=20)
        labels = clusterer.fit_predict(X_role_aug)
        inertia = float(getattr(clusterer, "inertia_", float("nan")))
        algo = "KMeans+RoleAug (fallback)"
    else:
        algo = "KMedoids" if _HAS_KMEDOIDS else "KMeans (fallback)"
        clusterer = _make_clusterer(n_clusters, cfg.random_seed)
        labels = clusterer.fit_predict(X_pca)
        inertia = float(getattr(clusterer, "inertia_", float("nan")))

    sil = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else 0.0
    log.info(
        "%s (K=%d): inertia=%.2f, silhouette=%.4f",
        algo, n_clusters, inertia, sil,
    )

    # ── Build output DataFrame ────────────────────────────────────────────────
    df_out = df.copy()
    df_out["cluster_id"] = labels
    pca_cols = [f"pca_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    df_out = pd.concat([df_out, pca_df], axis=1)

    return ClusterResult(
        df=df_out,
        cluster_labels=labels,
        pca_components=pca_df,
        inertia=inertia,
        silhouette=float(sil),
        explained_variance=pca.explained_variance_ratio_.tolist(),
        feature_names=pca_cols,
        n_clusters_used=n_clusters,
    )


def find_low_cost_alternatives(
    result: ClusterResult,
    top_percentile: float = 0.70,
    max_per_top_player: int = 5,
    cost_column: str = "team_rank_norm",
) -> list[LowCostAlternative]:
    """Identify under-valued players similar in profile to top performers.

    A player is considered "top" if their predicted/actual fantavoto_medio
    is above the ``top_percentile`` threshold.

    A "low-cost" alternative must:
    - Be in the same cluster as the top player.
    - Have a higher ``cost_column`` value (team_rank_norm: higher = smaller
      club = likely lower market price).
    - Not be a top-tier player themselves.

    Args:
        result: :class:`ClusterResult` from :func:`run_clustering`.
        top_percentile: Percentile threshold for "top player" label.
        max_per_top_player: Maximum number of alternatives per top player.
        cost_column: Proxy for player cost (higher = smaller club).

    Returns:
        List of :class:`LowCostAlternative` records, sorted by distance.
    """
    df = result.df.copy().reset_index(drop=True)

    rating_col = (
        "predicted_fantavoto" if "predicted_fantavoto" in df.columns
        else "fantavoto_medio"
    )
    if rating_col not in df.columns:
        raise ValueError("DataFrame must contain 'fantavoto_medio' or 'predicted_fantavoto'.")

    threshold = df[rating_col].quantile(top_percentile)
    is_top = df[rating_col] >= threshold

    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    X_pca = df[pca_cols].values

    alternatives: list[LowCostAlternative] = []

    for top_idx in df.index[is_top]:
        top_row = df.loc[top_idx]
        cluster = top_row["cluster_id"]
        top_vec = X_pca[top_idx]

        same_cluster = df["cluster_id"] == cluster
        not_top = ~is_top
        mask = same_cluster & not_top
        if "canonical_role" in df.columns:
            top_role = top_row.get("canonical_role")
            if top_role:
                mask = mask & (df["canonical_role"] == top_role)
        if cost_column in df.columns:
            top_cost = top_row.get(cost_column, np.nan)
            if not np.isnan(top_cost):
                mask = mask & (df[cost_column] > top_cost)

        candidate_indices = df.index[mask].tolist()
        if not candidate_indices:
            continue

        cand_vecs = X_pca[candidate_indices]
        dists = np.linalg.norm(cand_vecs - top_vec, axis=1)
        order = np.argsort(dists)

        for rank in order[:max_per_top_player]:
            cand_idx = candidate_indices[rank]
            cand_row = df.loc[cand_idx]
            alternatives.append(
                LowCostAlternative(
                    top_player_id=int(top_row.get("player_fotmob_id", -1)),
                    top_player_name=str(top_row.get("player_name", "")),
                    top_player_team=str(top_row.get("team_name", "")),
                    top_player_fantavoto=float(top_row[rating_col]),
                    alt_player_id=int(cand_row.get("player_fotmob_id", -1)),
                    alt_player_name=str(cand_row.get("player_name", "")),
                    alt_player_team=str(cand_row.get("team_name", "")),
                    alt_player_fantavoto=float(cand_row[rating_col]),
                    cluster_id=int(cluster),
                    distance=float(dists[rank]),
                )
            )

    alternatives.sort(key=lambda x: (x.top_player_name, x.distance))
    log.info("Found %d low-cost alternatives.", len(alternatives))
    return alternatives


# ── Visualisation ─────────────────────────────────────────────────────────────

def _draw_silhouette_panel(
    ax: plt.Axes,
    X_pca: np.ndarray,
    labels: np.ndarray,
    mean_sil: float,
) -> None:
    """Render per-cluster silhouette coefficient bars onto *ax*."""
    unique_clusters = sorted(set(labels))
    n_clusters = len(unique_clusters)
    sil_vals = silhouette_samples(X_pca, labels)

    y_lower = 10
    for i, cid in enumerate(unique_clusters):
        cluster_sil = np.sort(sil_vals[labels == cid])
        y_upper = y_lower + len(cluster_sil)
        color = plt.cm.tab10(float(i) / max(n_clusters, 1))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil,
            facecolor=color,
            alpha=0.75,
        )
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(cid), fontsize=8)
        y_lower = y_upper + 10

    ax.axvline(
        x=mean_sil,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {mean_sil:.3f}",
    )
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Analysis")
    ax.legend(fontsize=8)


def plot_clusters(
    result: ClusterResult,
    output_path: str,
    rating_col: str = "fantavoto_medio",
) -> None:
    """Save a combined PCA scatter + Silhouette plot.

    Left panel: 2-D PCA scatter coloured by cluster; top 10% by rating
    are annotated with player names.
    Right panel: Silhouette coefficient distribution per cluster.
    """
    df = result.df
    X_pca = result.pca_components.values
    labels = result.cluster_labels
    n_clusters = df["cluster_id"].nunique()

    fig, (ax_scatter, ax_sil) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: PCA scatter ─────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    color_map = {
        c: colors[i] for i, c in enumerate(sorted(df["cluster_id"].unique()))
    }

    for cluster_id, color in color_map.items():
        mask = df["cluster_id"] == cluster_id
        ax_scatter.scatter(
            df.loc[mask, "pca_0"],
            df.loc[mask, "pca_1"],
            c=[color],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=40,
            edgecolors="none",
        )

    if rating_col in df.columns:
        threshold = df[rating_col].quantile(0.90)
        for _, row in df[df[rating_col] >= threshold].iterrows():
            ax_scatter.annotate(
                row.get("player_name", ""),
                (row["pca_0"], row["pca_1"]),
                fontsize=6,
                alpha=0.8,
            )

    ax_scatter.set_xlabel(
        f"PCA Component 1 ({result.explained_variance[0]:.1%} var)"
    )
    ax_scatter.set_ylabel(
        f"PCA Component 2 ({result.explained_variance[1]:.1%} var)"
    )
    ax_scatter.set_title(
        f"Player Clusters — {result.n_clusters_used} clusters (PCA-reduced)"
    )
    ax_scatter.legend(loc="best", fontsize=8)

    # ── Right: Silhouette plot ────────────────────────────────────────────────
    if len(set(labels)) > 1:
        _draw_silhouette_panel(ax_sil, X_pca, labels, result.silhouette)
    else:
        ax_sil.text(
            0.5, 0.5,
            "Silhouette requires ≥ 2 clusters",
            ha="center", va="center",
            transform=ax_sil.transAxes,
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Cluster visualisation saved to %s", output_path)
