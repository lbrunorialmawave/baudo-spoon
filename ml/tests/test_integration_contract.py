"""Integration contract tests for Phase 0.

Risk-based tests verifying:
1. Player (V1) schema is unchanged
2. to_player_v1(PlayerV2) produces V1-compatible objects
3. MissingDataPolicy enum has all 4 required values
4. All 6 TargetSpec instances are constructable and have correct types
5. PredictionExplanation SHAP coherence invariant
6. Feature.safe_compute FAIL policy raises on missing columns
7. ScheduleAdjustmentConfig weight sum validation
"""

import polars as pl
import pytest

from ml.domain.config import (
    DEFAULT_SCHEDULE_ADJUSTMENT,
    ScheduleAdjustmentConfig,
)
from ml.domain.features import Feature, MissingDataPolicy
from ml.domain.player_versions import PlayerV1, PlayerV2, to_player_v1
from ml.domain.predictions import SHAP_TOLERANCE, PredictionExplanation
from ml.domain.targets import (
    BONUS_PREVISTI,
    FANTAPUNTI_TOTALI,
    FANTAVOTO_MEDIO,
    MINUTI_GIOCATI,
    PREZZO_ATTESO,
    PROBABILITA_TITOLARITA,
)
from ml.optimizer.models import Player

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_player_v1() -> Player:
    return Player(
        player_id="12345",
        name="Test Player",
        role="A",
        real_team="Juventus",
        cost=25,
        projected_score=6.8,
    )


@pytest.fixture
def sample_player_v2() -> PlayerV2:
    return PlayerV2(
        player_id="12345",
        name="Test Player",
        role="A",
        real_team="Juventus",
        cost=25,
        projected_score=6.8,
        expected_minutes=2700.0,
        var_value=0.4,
        expected_auction_price=28.0,
    )


@pytest.fixture
def coherent_explanation() -> PredictionExplanation:
    shap = {"feature_a": 0.3, "feature_b": -0.1}
    base = 6.5
    prediction = base + sum(shap.values())  # 6.7
    return PredictionExplanation(
        prediction=prediction,
        confidence=0.85,
        variance=0.02,
        prediction_interval=(6.2, 7.2),
        best_case=7.5,
        worst_case=5.5,
        top_features=[("feature_a", 0.3), ("feature_b", -0.1)],
        shap_values=shap,
        base_value=base,
    )


# ── Test 1: Player V1 schema unchanged ───────────────────────────────────────


class TestPlayerV1Schema:
    def test_required_fields_present(self, sample_player_v1: Player) -> None:
        assert sample_player_v1.player_id == "12345"
        assert sample_player_v1.name == "Test Player"
        assert sample_player_v1.role == "A"
        assert sample_player_v1.real_team == "Juventus"
        assert sample_player_v1.cost == 25
        assert sample_player_v1.projected_score == 6.8
        assert sample_player_v1.reliability_weight is None

    def test_role_validation(self) -> None:
        for valid_role in ("P", "D", "C", "A"):
            p = Player(
                player_id="x",
                name="n",
                role=valid_role,
                real_team="t",
                cost=1,
                projected_score=6.0,
            )  # type: ignore[arg-type]
            assert p.role == valid_role

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="role"):
            Player(
                player_id="x",
                name="n",
                role="GK",
                real_team="t",
                cost=1,
                projected_score=6.0,
            )  # type: ignore[arg-type]

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost"):
            Player(
                player_id="x",
                name="n",
                role="A",
                real_team="t",
                cost=-1,
                projected_score=6.0,
            )

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValueError, match="projected_score"):
            Player(
                player_id="x",
                name="n",
                role="A",
                real_team="t",
                cost=1,
                projected_score=-0.1,
            )

    def test_frozen(self, sample_player_v1: Player) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_player_v1.cost = 999  # type: ignore[misc]

    def test_hashable(self, sample_player_v1: Player) -> None:
        s = {sample_player_v1}
        assert sample_player_v1 in s


# ── Test 2: to_player_v1 adapter ─────────────────────────────────────────────


class TestPlayerVersionAdapter:
    def test_v2_to_v1_fields(self, sample_player_v2: PlayerV2) -> None:
        v1 = to_player_v1(sample_player_v2)
        assert isinstance(v1, Player)
        assert v1.player_id == sample_player_v2.player_id
        assert v1.name == sample_player_v2.name
        assert v1.role == sample_player_v2.role
        assert v1.real_team == sample_player_v2.real_team
        assert v1.cost == sample_player_v2.cost
        assert v1.projected_score == sample_player_v2.projected_score
        assert v1.reliability_weight == sample_player_v2.reliability_weight

    def test_v2_only_fields_stripped(self, sample_player_v2: PlayerV2) -> None:
        v1 = to_player_v1(sample_player_v2)
        assert not hasattr(v1, "expected_minutes")
        assert not hasattr(v1, "var_value")
        assert not hasattr(v1, "expected_auction_price")
        assert not hasattr(v1, "prediction_explanation")

    def test_v1_is_playerv1_type(self) -> None:
        assert PlayerV1 is Player

    def test_v2_validation_same_as_v1(self) -> None:
        with pytest.raises(ValueError):
            PlayerV2(
                player_id="",
                name="n",
                role="A",
                real_team="t",
                cost=1,
                projected_score=6.0,
            )
        with pytest.raises(ValueError):
            PlayerV2(
                player_id="x",
                name="n",
                role="GK",
                real_team="t",
                cost=1,
                projected_score=6.0,
            )  # type: ignore[arg-type]


# ── Test 3: MissingDataPolicy enum ───────────────────────────────────────────


class TestMissingDataPolicy:
    def test_all_four_values(self) -> None:
        values = {p.value for p in MissingDataPolicy}
        assert values == {"fail", "impute_role_median", "impute_zero", "proxy_feature"}

    def test_enum_members(self) -> None:
        assert MissingDataPolicy.FAIL.value == "fail"
        assert MissingDataPolicy.IMPUTE_ROLE_MEDIAN.value == "impute_role_median"
        assert MissingDataPolicy.IMPUTE_ZERO.value == "impute_zero"
        assert MissingDataPolicy.PROXY_FEATURE.value == "proxy_feature"


# ── Test 4: TargetSpec instances ─────────────────────────────────────────────


class TestTargetSpecs:
    ALL_SPECS = (
        FANTAVOTO_MEDIO,
        FANTAPUNTI_TOTALI,
        BONUS_PREVISTI,
        MINUTI_GIOCATI,
        PROBABILITA_TITOLARITA,
        PREZZO_ATTESO,
    )

    def test_six_targets_defined(self) -> None:
        assert len(self.ALL_SPECS) == 6

    def test_names_unique(self) -> None:
        names = [s.name for s in self.ALL_SPECS]
        assert len(names) == len(set(names))

    def test_target_types_valid(self) -> None:
        valid = {"regression", "classification", "probability"}
        for spec in self.ALL_SPECS:
            assert spec.target_type in valid, (
                f"{spec.name} has invalid type {spec.target_type}"
            )

    def test_fantavoto_medio_is_regression_no_transform(self) -> None:
        assert FANTAVOTO_MEDIO.target_type == "regression"
        assert FANTAVOTO_MEDIO.transform is None

    def test_probabilita_titolarita_is_probability(self) -> None:
        assert PROBABILITA_TITOLARITA.target_type == "probability"

    def test_log_transforms_roundtrip(self) -> None:
        for spec in (FANTAPUNTI_TOTALI, BONUS_PREVISTI, PREZZO_ATTESO):
            assert spec.transform is not None
            assert spec.inverse_transform is not None
            s = pl.Series([0.0, 1.0, 10.0, 100.0])
            transformed = spec.transform(s)
            recovered = spec.inverse_transform(transformed)
            for orig, rec in zip(s.to_list(), recovered.to_list()):
                assert abs(orig - rec) < 1e-9, (
                    f"Roundtrip failed for {spec.name}: {orig} -> {rec}"
                )

    def test_target_spec_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            FANTAVOTO_MEDIO.name = "other"  # type: ignore[misc]


# ── Test 5: PredictionExplanation SHAP coherence ─────────────────────────────


class TestPredictionExplanation:
    def test_coherent_explanation_passes(
        self, coherent_explanation: PredictionExplanation
    ) -> None:
        assert coherent_explanation.is_shap_coherent()
        assert coherent_explanation.shap_coherence_error() < SHAP_TOLERANCE

    def test_incoherent_explanation_detected(self) -> None:
        shap = {"feature_a": 0.3, "feature_b": -0.1}
        expl = PredictionExplanation(
            prediction=99.0,  # deliberately wrong
            confidence=0.5,
            variance=0.1,
            prediction_interval=(90.0, 100.0),
            best_case=100.0,
            worst_case=90.0,
            top_features=[],
            shap_values=shap,
            base_value=6.5,
        )
        assert not expl.is_shap_coherent()
        assert expl.shap_coherence_error() > SHAP_TOLERANCE

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_interval"):
            PredictionExplanation(
                prediction=6.5,
                confidence=0.5,
                variance=0.0,
                prediction_interval=(7.0, 6.0),  # lo > hi
                best_case=7.0,
                worst_case=6.0,
                top_features=[],
                shap_values={},
                base_value=6.5,
            )

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            PredictionExplanation(
                prediction=6.5,
                confidence=1.5,  # > 1.0
                variance=0.0,
                prediction_interval=(6.0, 7.0),
                best_case=7.0,
                worst_case=6.0,
                top_features=[],
                shap_values={},
                base_value=6.5,
            )

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="variance"):
            PredictionExplanation(
                prediction=6.5,
                confidence=0.5,
                variance=-0.1,
                prediction_interval=(6.0, 7.0),
                best_case=7.0,
                worst_case=6.0,
                top_features=[],
                shap_values={},
                base_value=6.5,
            )


# ── Test 6: Feature.safe_compute FAIL policy ─────────────────────────────────


class TestFeatureSafeCompute:
    def test_fail_policy_raises_on_missing_column(self) -> None:
        class DummyFeature(Feature):
            name = "test_feature"
            required_columns = frozenset(["goals_per90", "xg_per90"])
            missing_data_policy = MissingDataPolicy.FAIL

            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["goals_per90"] + data["xg_per90"]

        feat = DummyFeature()
        df = pl.DataFrame({"goals_per90": [1.0, 2.0]})  # missing xg_per90
        with pytest.raises(ValueError, match="xg_per90"):
            feat.safe_compute(df)

    def test_fail_policy_succeeds_when_all_columns_present(self) -> None:
        class DummyFeature(Feature):
            name = "test_feature"
            required_columns = frozenset(["goals_per90"])
            missing_data_policy = MissingDataPolicy.FAIL

            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["goals_per90"] * 2.0

        feat = DummyFeature()
        df = pl.DataFrame({"goals_per90": [1.0, 2.0]})
        result = feat.safe_compute(df)
        assert result.to_list() == [2.0, 4.0]

    def test_impute_zero_policy_fills_missing(self) -> None:
        class DummyFeature(Feature):
            name = "test_feature"
            required_columns = frozenset(["goals_per90", "xg_per90"])
            missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["goals_per90"] + data["xg_per90"]

        feat = DummyFeature()
        df = pl.DataFrame({"goals_per90": [1.0, 2.0]})  # xg_per90 missing
        result = feat.safe_compute(df)
        # xg_per90 filled with 0.0
        assert result.to_list() == [1.0, 2.0]

    def test_proxy_feature_name_required_for_proxy_policy(self) -> None:
        with pytest.raises(TypeError, match="proxy_feature_name"):

            class BadFeature(Feature):
                name = "bad"
                required_columns = frozenset(["xg_per90"])
                missing_data_policy = MissingDataPolicy.PROXY_FEATURE
                # proxy_feature_name NOT set — should raise TypeError in __init_subclass__

                def compute(self, data: pl.DataFrame) -> pl.Series:
                    return data["xg_per90"]

    def test_proxy_feature_substitutes_column(self) -> None:
        class DummyFeature(Feature):
            name = "test_feature"
            required_columns = frozenset(["xg_per90"])
            missing_data_policy = MissingDataPolicy.PROXY_FEATURE
            proxy_feature_name = "total_scoring_att_per90"

            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["xg_per90"] * 0.5

        feat = DummyFeature()
        df = pl.DataFrame({"total_scoring_att_per90": [4.0, 6.0]})  # xg_per90 missing
        result = feat.safe_compute(df)
        assert result.to_list() == [2.0, 3.0]


# ── Test 7: ScheduleAdjustmentConfig validation ───────────────────────────────


class TestScheduleAdjustmentConfig:
    def test_default_weights_sum_to_one(self) -> None:
        cfg = DEFAULT_SCHEDULE_ADJUSTMENT
        total = (
            cfg.elo_weight
            + cfg.expected_points_weight
            + cfg.league_position_weight
            + cfg.goal_difference_weight
            + cfg.squad_value_weight
        )
        assert abs(total - 1.0) < 1e-9

    def test_invalid_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScheduleAdjustmentConfig(elo_weight=0.5)  # sum != 1.0

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="coeff_min"):
            ScheduleAdjustmentConfig(
                coeff_min=1.5,
                coeff_max=0.5,
                elo_weight=0.30,
                expected_points_weight=0.25,
                league_position_weight=0.20,
                goal_difference_weight=0.15,
                squad_value_weight=0.10,
            )

    def test_default_coeff_range(self) -> None:
        assert DEFAULT_SCHEDULE_ADJUSTMENT.coeff_min == 0.7
        assert DEFAULT_SCHEDULE_ADJUSTMENT.coeff_max == 1.3
