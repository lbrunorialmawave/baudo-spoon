"""Unit tests for the offline experiment harness (PR5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.config import MLConfig
from ml.experiments import (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    ExperimentVariant,
    apply_variant,
    default_variants,
)


def _base_cfg(tmp_path: Path) -> MLConfig:
    """Build a minimal ``MLConfig`` for testing the harness in isolation."""
    return MLConfig(artifacts_dir=tmp_path, test_seasons=1)


class TestDefaultVariants:
    def test_canonical_matrix_has_four_variants(self) -> None:
        variants = default_variants()
        assert set(variants.keys()) == {VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D}

    def test_control_variant_disables_all_flags(self) -> None:
        v = default_variants()[VARIANT_A]
        assert v.enable_limited_sample_training is False
        assert v.enable_shrinkage is False
        assert v.enable_recent_role_features is False

    def test_progression_each_variant_turns_on_one_more_flag(self) -> None:
        v = default_variants()
        # B enables weighting, C adds shrinkage, D adds role features.
        assert v[VARIANT_B].enable_limited_sample_training is True
        assert v[VARIANT_C].enable_shrinkage is True
        assert v[VARIANT_D].enable_recent_role_features is True


class TestApplyVariant:
    def test_does_not_mutate_input(self, tmp_path: Path) -> None:
        base = _base_cfg(tmp_path)
        before = base.model_dump()
        v = default_variants()[VARIANT_C]
        apply_variant(base, v)
        # No mutation
        assert base.model_dump() == before

    def test_always_disables_breakout_model(self, tmp_path: Path) -> None:
        base = _base_cfg(tmp_path).model_copy(update={"enable_breakout_model": True})
        v = default_variants()[VARIANT_D]
        cfg = apply_variant(base, v)
        assert cfg.enable_breakout_model is False

    def test_strategy_overridden(self, tmp_path: Path) -> None:
        base = _base_cfg(tmp_path)
        v = ExperimentVariant(
            name="custom",
            description="custom test variant",
            enable_limited_sample_training=True,
            enable_shrinkage=False,
            weighting_strategy="bucketed",
            enable_recent_role_features=False,
        )
        cfg = apply_variant(base, v)
        assert cfg.weighting_strategy == "bucketed"
        assert cfg.enable_limited_sample_training is True


class TestRunExperiment:
    def test_report_is_persisted(self, tmp_path: Path) -> None:
        base = _base_cfg(tmp_path)
        # Provide a minimal pre-built fake trainer output to keep the test
        # fast and hermetic (no DB / no training).  We do this by stubbing
        # the Trainer class only for this test.
        from ml.experiments import harness as harness_mod

        def _fake_trainer(cfg):  # type: ignore[no-untyped-def]
            class _StubTrainer:
                def run(self, external_fantavoto_csv=None):  # type: ignore[no-untyped-def]
                    return {
                        "best_model": "ridge",
                        "role_metrics": {
                            "outfield": {
                                "ridge": {"rmse": 0.40, "mae": 0.30, "r2": 0.20},
                            },
                            "gk": {
                                "ridge": {"rmse": 0.50, "mae": 0.40, "r2": 0.10},
                            },
                        },
                        "backtest": {"mean_rmse": 0.45, "mean_mae": 0.35},
                        "sample_reliability": {
                            "cohort_profile": {
                                "n_total": 1000, "n_limited": 200, "n_standard": 800,
                                "n_insufficient": 0,
                            }
                        },
                    }
            return _StubTrainer()

        original = harness_mod.Trainer
        harness_mod.Trainer = _fake_trainer  # type: ignore[assignment]
        try:
            report = harness_mod.run_experiment(base)
        finally:
            harness_mod.Trainer = original  # type: ignore[assignment]

        report_path = tmp_path / "experiments" / report["run_id"] / "report.json"
        assert report_path.is_file()
        on_disk = json.loads(report_path.read_text(encoding="utf-8"))
        assert on_disk["run_id"] == report["run_id"]
        for name in (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D):
            assert on_disk["variants"][name]["status"] == "ok"
            assert on_disk["variants"][name]["rmse"] == pytest.approx(0.40)

    def test_failed_variant_is_captured_not_raised(self, tmp_path: Path) -> None:
        base = _base_cfg(tmp_path)
        from ml.experiments import harness as harness_mod

        def _exploding_trainer(cfg):  # type: ignore[no-untyped-def]
            class _Exploder:
                def run(self, external_fantavoto_csv=None):  # type: ignore[no-untyped-def]
                    raise RuntimeError("simulated failure")
            return _Exploder()

        original = harness_mod.Trainer
        harness_mod.Trainer = _exploding_trainer  # type: ignore[assignment]
        try:
            report = harness_mod.run_experiment(base, variants={VARIANT_A: default_variants()[VARIANT_A]})
        finally:
            harness_mod.Trainer = original  # type: ignore[assignment]

        assert report["variants"][VARIANT_A]["status"] == "error"
        assert "simulated failure" in report["variants"][VARIANT_A]["error"]
