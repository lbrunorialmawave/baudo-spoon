"""Tests for high-cost FWD role-default warning (P4)."""

from __future__ import annotations

import logging

import pandas as pd

from ml.data.loader import _warn_high_cost_role_defaults


def test_warn_high_cost_role_defaults_emits_warning(caplog):
    df = pd.DataFrame([
        {"player_name": "Cheap Filler", "team": "Empoli", "qt_a": 1, "canonical_role": "FWD"},
        {"player_name": "Expensive Neo", "team": "Inter", "qt_a": 25, "canonical_role": None},
        {"player_name": "Mid Neo", "team": "Atalanta", "qt_a": 12, "canonical_role": None},
    ])
    missing = df["canonical_role"].isna()
    df["canonical_role"] = df["canonical_role"].fillna("FWD")

    with caplog.at_level(logging.WARNING):
        _warn_high_cost_role_defaults(df, missing, logging.getLogger("test"))

    assert any("Role defaulted to 'FWD'" in r.message for r in caplog.records)
    assert any("Expensive Neo" in r.message for r in caplog.records)


def test_warn_no_op_when_mask_empty(caplog):
    df = pd.DataFrame([
        {"player_name": "OK", "team": "Roma", "qt_a": 10, "canonical_role": "MID"},
    ])
    missing = df["canonical_role"].isna()

    with caplog.at_level(logging.WARNING):
        _warn_high_cost_role_defaults(df, missing, logging.getLogger("test"))

    assert not any("Role defaulted" in r.message for r in caplog.records)
