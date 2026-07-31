"""Task 0 (option a): confirm ``min_start_probability`` and ``replacement_method``
are restored to ``OptimizationRequest`` (mirror of the fields already present on
``AuctionConfigSchema``) and retain the documented defaults.

This is the inverse of a prior test that asserted the fields were removed from
the optimizer's request schema (the removal left the frontend controls on the
Optimizer screen silently dead). With option (a) the optimizer pool-building
path and the ``VarEngine`` used for ESV/VAR blending must accept the same two
parameters that the auction router already supports.
"""

from __future__ import annotations

from api.src.schemas import OptimizationRequest


def test_min_start_probability_field_restored() -> None:
    """``OptimizationRequest`` exposes ``min_start_probability`` (Optional)."""
    assert "min_start_probability" in OptimizationRequest.model_fields
    field = OptimizationRequest.model_fields["min_start_probability"]
    # Default must be ``None`` (preserves legacy behaviour: no pre-filter).
    assert field.default is None


def test_replacement_method_field_restored() -> None:
    """``OptimizationRequest`` exposes ``replacement_method`` (default percentile)."""
    assert "replacement_method" in OptimizationRequest.model_fields
    field = OptimizationRequest.model_fields["replacement_method"]
    # Default must be ``"percentile"`` (matches the existing VarEngine default
    # and the auction's contract).
    assert field.default == "percentile"


def test_auction_config_still_has_fields() -> None:
    """``AuctionConfigSchema`` retains the same two fields (regression guard)."""
    from api.src.schemas import AuctionConfigSchema

    assert "min_start_probability" in AuctionConfigSchema.model_fields
    assert "replacement_method" in AuctionConfigSchema.model_fields
