"""Tests for ml.scripts.check_promotion_gate (WS4.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.scripts.check_promotion_gate import _check_variant, main


def _report(**variant_overrides) -> dict:
    base_variant = {
        "status": "ok",
        "mae": 0.30,
        "rmse": 0.40,
        "mae_by_cohort": {"STANDARD": 0.28, "LIMITED": 0.35, "INSUFFICIENT": None},
        "rmse_by_cohort": {"STANDARD": 0.38, "LIMITED": 0.45, "INSUFFICIENT": None},
        "phenom_leakage_rate": 0.10,
    }
    base_variant.update(variant_overrides)
    return {
        "run_id": "test-run",
        "variants": {
            "A_control": {
                "status": "ok",
                "mae": 0.30,
                "rmse": 0.40,
            },
            "C_shrinkage": base_variant,
        },
    }


def test_gate_passes_on_healthy_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    assert main([str(path)]) == 0


def test_gate_fails_on_high_phenom_leakage(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(phenom_leakage_rate=0.80)), encoding="utf-8")
    assert main([str(path)]) == 1


def test_gate_fails_on_mae_regression(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(mae=0.50)), encoding="utf-8")  # +66% vs 0.30
    assert main([str(path), "--max-mae-delta-pct", "3.0"]) == 1


def test_check_variant_missing_keys() -> None:
    report = {
        "variants": {
            "C_shrinkage": {"status": "ok", "mae": 0.3},
            "A_control": {"status": "ok", "mae": 0.3},
        }
    }
    failures = _check_variant(
        report,
        variant="C_shrinkage",
        control="A_control",
        max_phenom_leakage=0.25,
        max_mae_delta_pct=3.0,
        require_cohort_keys=True,
    )
    assert any("mae_by_cohort" in f for f in failures)
