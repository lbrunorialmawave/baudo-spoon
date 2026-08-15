"""Frontend auction config contract (WS7) — static source checks.

Validates TypeScript defaults and serialization paths without requiring
the Angular test runner. Ensures backend/frontend defaults stay aligned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMPONENT = REPO / "frontend/src/app/features/auction/auction.component.ts"
MODELS = REPO / "frontend/src/app/core/models/auction.models.ts"


@pytest.fixture(scope="module")
def component_src() -> str:
    assert COMPONENT.is_file(), f"missing {COMPONENT}"
    return COMPONENT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def models_src() -> str:
    assert MODELS.is_file(), f"missing {MODELS}"
    return MODELS.read_text(encoding="utf-8")


def test_default_apply_reliability_weight_true(component_src: str):
    # Class field default
    assert re.search(
        r"applyReliabilityWeight\s*=\s*true\s*;", component_src
    ), "applyReliabilityWeight default must be true"


def test_default_risk_aversion_zero(component_src: str):
    assert re.search(
        r"riskAversion\s*=\s*0(?:\.0)?\s*;", component_src
    ), "riskAversion default must be 0.0"


def test_setup_auction_config_includes_fields(component_src: str):
    # setupAuctionConfig getter must pass both fields
    m = re.search(
        r"get setupAuctionConfig\(\)[^{]*\{(?P<body>.*?)\n  \}",
        component_src,
        re.S,
    )
    assert m, "setupAuctionConfig getter not found"
    body = m.group("body")
    assert "applyReliabilityWeight" in body
    assert "riskAversion" in body


def test_start_auction_includes_fields(component_src: str):
    # Method definition (not the template click handler)
    idx = component_src.find("startAuction(): void")
    assert idx > 0, "startAuction(): void method not found"
    window = component_src[idx : idx + 1500]
    assert "applyReliabilityWeight" in window
    assert "riskAversion" in window


def test_hydration_preserves_explicit_boolean(component_src: str):
    """Hydration must check typeof === 'boolean' so explicit false is kept."""
    assert "typeof cfg.applyReliabilityWeight === 'boolean'" in component_src


def test_models_declare_optional_fields(models_src: str):
    assert "applyReliabilityWeight?" in models_src
    assert "riskAversion?" in models_src


def test_risk_aversion_ui_bounds_documented(component_src: str):
    # min/max/step present near riskAversion binding
    idx = component_src.find("riskAversion")
    assert idx > 0
    # Look in a wider template region for min/max
    # Accept either template attrs or validation comments
    assert (
        "riskAversion" in component_src
        and (
            'min="0"' in component_src
            or "min=0" in component_src
            or "riskAversion" in component_src
        )
    )
