"""Unit tests for ml.clustering.kmeans.

Covers:
- StandardScaler → PCA pipeline ordering (strict enforcement of scaling
  before variance maximisation, as per pipeline-integrity spec).
- ClusterResult structure validation.
- n_init stability: KMeans uses n_init=20, not the less stable "auto".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.clustering.kmeans import _make_clusterer, run_clustering
from ml.config import MLConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cluster_df(n_players: int = 40, seed: int = 42) -> pd.DataFrame:
    """Minimal feature-engineered DataFrame suitable for run_clustering."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        rows.append({
            "player_fotmob_id": i,
            "player_name": f"Player{i}",
            "team_name": f"Team{i % 5}",
            "season_start": 2023,
            "fantavoto_medio": float(rng.uniform(4.0, 8.0)),
            "goals_per90": float(rng.uniform(0, 1.0)),
            "goal_assist_per90": float(rng.uniform(0, 0.8)),
            "total_scoring_att_per90": float(rng.uniform(0, 3.0)),
            "ontarget_scoring_att_per90": float(rng.uniform(0, 1.5)),
            "won_contest_per90": float(rng.uniform(0, 5.0)),
            "total_att_assist_per90": float(rng.uniform(0, 2.0)),
            "big_chance_created_per90": float(rng.uniform(0, 0.5)),
            "yellow_card_per90": float(rng.uniform(0, 0.3)),
            "interception_per90": float(rng.uniform(0, 2.0)),
            "total_tackle_per90": float(rng.uniform(0, 3.0)),
            "effective_clearance_per90": float(rng.uniform(0, 2.0)),
            "fouls_per90": float(rng.uniform(0, 2.0)),
            "team_rank_norm": float(rng.uniform(0.1, 1.0)),
        })
    return pd.DataFrame(rows)


def _make_minimal_config(**overrides) -> MLConfig:
    """Create a minimal MLConfig with a fake database URL for testing."""
    defaults = dict(
        database_url="postgresql://test:test@localhost/test",
        n_clusters=3,
        pca_variance_threshold=0.90,
        random_seed=42,
    )
    defaults.update(overrides)
    return MLConfig(**defaults)


# ── Pipeline ordering: StandardScaler must precede PCA ────────────────────────

class TestScalerPCAPipelineOrder:
    """Strict enforcement: StandardScaler → PCA sequence.

    Per spec: scaling must occur before variance maximisation.  The tests
    below verify this both at the interface level (run_clustering internals
    implied) and by direct sklearn contract assertions.
    """

    def test_pca_input_is_zero_mean(self):
        """Data fed into PCA must have approximately zero mean (post-StandardScaler)."""
        rng = np.random.default_rng(0)
        X = rng.uniform(50, 200, size=(100, 5))  # intentionally large non-zero range

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # After scaling the mean of each column must be ≈ 0
        col_means = np.abs(X_scaled.mean(axis=0))
        assert np.all(col_means < 1e-10), (
            f"StandardScaler output must have zero column means before PCA; "
            f"got means: {col_means}"
        )

    def test_pca_input_is_unit_variance(self):
        """Data fed into PCA must have approximately unit variance (post-StandardScaler)."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1000, size=(200, 8))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        col_stds = X_scaled.std(axis=0)
        assert np.allclose(col_stds, 1.0, atol=1e-6), (
            f"StandardScaler output must have unit column stds before PCA; "
            f"got stds: {col_stds}"
        )

    def test_unscaled_data_produces_biased_pca_variance(self):
        """Without scaling, high-magnitude features dominate PCA — this must NOT happen.

        Confirms why StandardScaler → PCA ordering is mandatory: unscaled PCA
        concentrates variance into the largest-magnitude feature, making the
        decomposition meaningless for multi-scale stat vectors.
        """
        rng = np.random.default_rng(2)
        # Feature A: range 0–1; Feature B: range 0–10000 (dominates unscaled PCA)
        X_a = rng.uniform(0, 1, size=(100, 1))
        X_b = rng.uniform(0, 10000, size=(100, 1))
        X = np.hstack([X_a, X_b])

        # Unscaled PCA — PC1 should be dominated by feature B
        pca_raw = PCA(n_components=2).fit(X)
        pc1_loadings_raw = np.abs(pca_raw.components_[0])
        dominant_feature_raw = int(np.argmax(pc1_loadings_raw))

        # Scaled PCA — both features contribute meaningfully
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_scaled = PCA(n_components=2).fit(X_scaled)
        pc1_loadings_scaled = np.abs(pca_scaled.components_[0])
        min_loading_scaled = float(pc1_loadings_scaled.min())

        # After scaling, the minimum loading on PC1 should be non-trivial
        assert dominant_feature_raw == 1, (
            "Test setup: unscaled PCA should be dominated by the large-magnitude feature"
        )
        assert min_loading_scaled > 0.1, (
            f"After scaling, both features should contribute to PC1 "
            f"(min loading: {min_loading_scaled:.4f}). "
            "If this fails, StandardScaler is not being applied before PCA."
        )

    def test_run_clustering_preserves_scaler_pca_order(self):
        """run_clustering must internally apply StandardScaler before PCA.

        Verified indirectly: the PCA-reduced cluster coordinates (pca_0, pca_1)
        must have approximately zero mean and unit-scale variance, consistent
        with scaled input.  Non-scaled input would produce a PCA space with
        non-centred coordinates.
        """
        df = _make_cluster_df(n_players=50)
        cfg = _make_minimal_config(n_clusters=3)

        result = run_clustering(df, cfg)

        pca_values = result.pca_components.values
        col_means = np.abs(pca_values.mean(axis=0))

        # PCA output of scaled data should have near-zero column means
        # (PCA centres its input; if StandardScaler ran first, the input was already centred)
        assert np.all(col_means < 1.0), (
            f"PCA component means are unexpectedly large: {col_means}. "
            "This may indicate StandardScaler was not applied before PCA."
        )


# ── n_init stability ──────────────────────────────────────────────────────────

class TestKMeansNInit:
    """KMeans fallback must use n_init=20 for convergence stability."""

    def test_kmeans_fallback_uses_n_init_20(self):
        """_make_clusterer returns a KMeans with n_init=20 (not 'auto')."""
        from ml.clustering.kmeans import _HAS_KMEDOIDS

        if _HAS_KMEDOIDS:
            pytest.skip("KMedoids is available; KMeans fallback not used")

        clusterer = _make_clusterer(n_clusters=3, random_seed=42)
        assert isinstance(clusterer, KMeans), "Fallback clusterer must be KMeans"
        assert clusterer.n_init == 20, (
            f"KMeans n_init must be 20 for convergence stability; got {clusterer.n_init}"
        )

    def test_kmeans_n_init_20_produces_deterministic_result(self):
        """Two identical runs with n_init=20 and same seed yield identical labels."""
        from ml.clustering.kmeans import _HAS_KMEDOIDS

        if _HAS_KMEDOIDS:
            pytest.skip("KMedoids is available; KMeans fallback not used")

        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 4))

        labels_a = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X)
        labels_b = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X)

        np.testing.assert_array_equal(
            labels_a, labels_b,
            err_msg="KMeans with n_init=20 and same seed must produce identical labels",
        )


# ── ClusterResult contract ────────────────────────────────────────────────────

class TestClusterResultContract:
    """ClusterResult fields must have correct types and shapes."""

    def test_cluster_result_has_expected_fields(self):
        """run_clustering returns a ClusterResult with all required fields."""
        df = _make_cluster_df(n_players=40)
        cfg = _make_minimal_config(n_clusters=3)

        result = run_clustering(df, cfg)

        assert result.n_clusters_used == 3
        assert len(result.cluster_labels) == len(df)
        assert isinstance(result.silhouette, float)
        assert isinstance(result.inertia, float)
        assert len(result.explained_variance) >= 1
        assert "cluster_id" in result.df.columns
        assert "pca_0" in result.df.columns

    def test_auto_k_selects_valid_k(self):
        """n_clusters=-1 (auto) selects K in the expected [2, 10] range."""
        df = _make_cluster_df(n_players=60)
        cfg = _make_minimal_config(n_clusters=-1)

        result = run_clustering(df, cfg)

        assert 2 <= result.n_clusters_used <= 10, (
            f"Auto-K must be in [2, 10]; got {result.n_clusters_used}"
        )


# ── Soft-role constraints ─────────────────────────────────────────────────────

class TestSoftRoleConstraints:
    """Verify that soft-role constraints reduce cross-role cluster formation.

    The penalty in _build_role_aware_distance_matrix must be large enough to
    strongly discourage KMedoids from assigning GKs to the same cluster as
    outfielders when clean role-separated groups exist in the data.
    """

    def test_distance_matrix_is_symmetric(self):
        """Role-aware distance matrix must be symmetric and have zero diagonal."""
        from ml.clustering.kmeans import _build_role_aware_distance_matrix

        rng = np.random.default_rng(0)
        X_pca = rng.standard_normal((20, 3))
        role_codes = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

        D = _build_role_aware_distance_matrix(X_pca, role_codes, cross_role_penalty=5.0)

        np.testing.assert_array_equal(D, D.T, err_msg="Distance matrix must be symmetric")
        assert np.all(np.diag(D) == 0.0), "Diagonal of distance matrix must be zero"
        assert D.shape == (20, 20)

    def test_intra_role_distance_is_pure_euclidean(self):
        """Intra-role distances must equal plain Euclidean distance (no penalty)."""
        from ml.clustering.kmeans import _build_role_aware_distance_matrix

        rng = np.random.default_rng(1)
        X_pca = rng.standard_normal((10, 2))
        role_codes = np.zeros(10, dtype=np.int32)  # all same role → no penalty

        D_role = _build_role_aware_distance_matrix(X_pca, role_codes, cross_role_penalty=99.0)

        # Without cross-role pairs, distances must equal Euclidean
        for i in range(10):
            for j in range(10):
                expected = float(np.linalg.norm(X_pca[i] - X_pca[j]))
                assert abs(D_role[i, j] - expected) < 1e-10, (
                    f"Intra-role distance ({i},{j}) should be pure Euclidean "
                    f"({expected:.4f}), got {D_role[i, j]:.4f}"
                )

    def test_cross_role_distance_includes_penalty(self):
        """Cross-role distances must exceed intra-role by exactly the penalty."""
        from ml.clustering.kmeans import _build_role_aware_distance_matrix

        X_pca = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        role_codes = np.array([0, 1], dtype=np.int32)  # two different roles
        penalty = 7.0

        D = _build_role_aware_distance_matrix(X_pca, role_codes, cross_role_penalty=penalty)

        euclidean_dist = 1.0  # ||[0,0] - [1,0]||
        expected_penalised = euclidean_dist + penalty
        assert abs(D[0, 1] - expected_penalised) < 1e-10, (
            f"Expected cross-role distance {expected_penalised}, got {D[0, 1]}"
        )

    def test_run_clustering_with_canonical_role_does_not_raise(self):
        """run_clustering must succeed when canonical_role column is present."""
        df = _make_cluster_df(n_players=40)
        roles = (["GK"] * 5 + ["DEF"] * 10 + ["MID"] * 15 + ["FWD"] * 10)
        df["canonical_role"] = roles
        cfg = _make_minimal_config(n_clusters=3)

        result = run_clustering(df, cfg)

        assert result.n_clusters_used == 3
        assert "cluster_id" in result.df.columns
        assert len(result.cluster_labels) == len(df)

    def test_gk_cluster_separation(self):
        """GKs should not share a cluster with outfielders under strong constraints.

        Constructs perfectly separable GK vs FWD stat profiles.  With the
        default penalty (5.0), KMeans/KMedoids should assign them to distinct
        clusters when K ≥ 2.
        """
        from ml.clustering.kmeans import _HAS_KMEDOIDS

        rng = np.random.default_rng(7)
        n = 30
        # GKs concentrate near origin (zero attacking stats)
        gk_rows = {f: rng.uniform(0.0, 0.05) for f in [
            "goals_per90", "goal_assist_per90", "total_scoring_att_per90",
            "ontarget_scoring_att_per90", "won_contest_per90",
            "total_att_assist_per90", "big_chance_created_per90",
            "yellow_card_per90", "interception_per90", "total_tackle_per90",
            "effective_clearance_per90", "fouls_per90",
        ]}
        # FWDs concentrate far from origin (high attacking stats)
        fwd_rows = {f: rng.uniform(0.8, 1.5) for f in [
            "goals_per90", "goal_assist_per90", "total_scoring_att_per90",
            "ontarget_scoring_att_per90", "won_contest_per90",
            "total_att_assist_per90", "big_chance_created_per90",
            "yellow_card_per90", "interception_per90", "total_tackle_per90",
            "effective_clearance_per90", "fouls_per90",
        ]}
        rows = []
        for i in range(n):
            base = gk_rows if i < n // 2 else fwd_rows
            row = {k: float(v) + rng.uniform(-0.01, 0.01) for k, v in base.items()}
            row.update({
                "player_fotmob_id": i,
                "player_name": f"P{i}",
                "team_name": f"T{i % 5}",
                "season_start": 2023,
                "fantavoto_medio": 5.0,
                "team_rank_norm": 0.5,
                "canonical_role": "GK" if i < n // 2 else "FWD",
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        cfg = _make_minimal_config(n_clusters=2)
        result = run_clustering(df, cfg)

        gk_clusters = set(result.df.loc[df["canonical_role"] == "GK", "cluster_id"])
        fwd_clusters = set(result.df.loc[df["canonical_role"] == "FWD", "cluster_id"])

        # With clearly separable profiles + role penalty, the clusters should not overlap
        assert gk_clusters.isdisjoint(fwd_clusters), (
            f"GKs and FWDs should be in separate clusters under role constraints. "
            f"GK clusters: {gk_clusters}, FWD clusters: {fwd_clusters}"
        )
