"""Task 3: confirm min_start_probability and replacement_method are removed from OptimizationRequest."""

from __future__ import annotations

from api.src.schemas import OptimizationRequest


def test_no_min_start_probability_field() -> None:
    """OptimizationRequest no longer accepts min_start_probability."""
    assert "min_start_probability" not in OptimizationRequest.model_fields


def test_no_replacement_method_field() -> None:
    """OptimizationRequest no longer accepts replacement_method."""
    assert "replacement_method" not in OptimizationRequest.model_fields


def test_auction_config_still_has_fields() -> None:
    """AuctionConfigSchema retains these fields (they belong there)."""
    from api.src.schemas import AuctionConfigSchema

    assert "min_start_probability" in AuctionConfigSchema.model_fields
    assert "replacement_method" in AuctionConfigSchema.model_fields
