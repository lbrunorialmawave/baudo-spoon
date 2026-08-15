"""Tests for the new ``--emit-effective-config`` / ``--emit-canary-report``
CLI flags in :mod:`ml.run_pipeline` (WS14, plan §16.1).

These flags were added because ``ml-training.yml`` (Phase 5 "Train")
expects the trainer to write the two artefacts directly.  Argparse
must accept them without complaining — the original bug was
``unrecognized arguments: --emit-effective-config …``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Database URL is required by pydantic-settings when the module is
# imported.  We never connect — we only exercise the CLI parser.
os.environ.setdefault(
    "ML_DATABASE_URL",
    "postgresql+psycopg2://fake:fake@localhost:5432/fake",
)

from ml.run_pipeline import _parse_args  # noqa: E402


def _relay_argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ml.run_pipeline", *args])


# ── Acceptance — the bug that triggered this change ────────────────────────


def test_emit_effective_config_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _relay_argv(
        monkeypatch,
        "--emit-effective-config",
        "artifacts/effective_config.json",
    )
    ns = _parse_args()
    assert ns.emit_effective_config == "artifacts/effective_config.json"


def test_emit_canary_report_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _relay_argv(
        monkeypatch,
        "--emit-canary-report",
        "artifacts/canary_report.json",
    )
    ns = _parse_args()
    assert ns.emit_canary_report == "artifacts/canary_report.json"


def test_both_flags_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    _relay_argv(
        monkeypatch,
        "--emit-effective-config",
        "a/eff.json",
        "--emit-canary-report",
        "a/canary.json",
    )
    ns = _parse_args()
    assert ns.emit_effective_config == "a/eff.json"
    assert ns.emit_canary_report == "a/canary.json"


# ── Default behaviour preserved ─────────────────────────────────────────────


def test_defaults_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _relay_argv(monkeypatch)
    ns = _parse_args()
    assert ns.emit_effective_config is None
    assert ns.emit_canary_report is None


# ── Regression: all pre-existing flags still parse ─────────────────────────


def test_existing_flags_still_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    _relay_argv(
        monkeypatch,
        "--league",
        "Serie A",
        "--clusters",
        "-1",
        "--tune-iter",
        "120",
        "--test-seasons",
        "1",
        "--min-minutes",
        "800",
        "--seed",
        "42",
        "--log-level",
        "INFO",
        "--tune",
        "--predict-next",
        "--fantavoto-csv",
        "/app/artifacts/fantavoto_real.csv",
        "--json-logs",
        "--emit-effective-config",
        "artifacts/effective_config.json",
        "--emit-canary-report",
        "artifacts/canary_report.json",
    )
    ns = _parse_args()
    # Spot-check each flag, focusing on the new + critical ones.
    assert ns.league == "Serie A"
    assert ns.clusters == -1
    assert ns.tune_iter == 120
    assert ns.test_seasons == 1
    assert ns.min_minutes == 800
    assert ns.seed == 42
    assert ns.log_level == "INFO"
    assert ns.tune is True
    assert ns.predict_next is True
    assert ns.fantavoto_csv == "/app/artifacts/fantavoto_real.csv"
    assert ns.json_logs is True
    assert Path(ns.emit_effective_config).name == "effective_config.json"
    assert Path(ns.emit_canary_report).name == "canary_report.json"
