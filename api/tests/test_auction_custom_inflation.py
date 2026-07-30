"""Task 4: AuctionConfigSchema accepts custom inflation_config and it affects pricing."""

from __future__ import annotations

from ml.auction.models import AuctionConfig
from ml.auction.price_drift import compute_baseline_cost
from ml.optimizer.models import InflationConfig, Player


def _player() -> Player:
    return Player(
        player_id="p1",
        name="Test",
        role="A",
        real_team="Inter",
        cost=20,
        projected_score=8.0,
    )


def test_custom_inflation_changes_baseline() -> None:
    """Custom inflation config produces different baseline than default."""
    default_cfg = AuctionConfig(
        num_participants=10,
        use_inflation_baseline=True,
        inflation_config=InflationConfig(),
    )
    custom_cfg = AuctionConfig(
        num_participants=10,
        use_inflation_baseline=True,
        inflation_config=InflationConfig(
            max_inflation_multiplier=3.0,
            base_inflation_rate=0.15,
        ),
    )
    p = _player()

    baseline_default = compute_baseline_cost(p, 0.95, default_cfg)
    baseline_custom = compute_baseline_cost(p, 0.95, custom_cfg)

    assert baseline_custom > baseline_default


def test_schema_accepts_inflation_config() -> None:
    """AuctionConfigSchema has optional inflation_config field."""
    from api.src.schemas import AuctionConfigSchema, InflationConfigSchema

    schema = AuctionConfigSchema(
        num_participants=8,
        use_inflation_baseline=True,
        inflation_config=InflationConfigSchema(max_inflation_multiplier=2.5),
    )
    assert schema.inflation_config is not None
    assert schema.inflation_config.max_inflation_multiplier == 2.5


def test_schema_none_inflation_backward_compat() -> None:
    """When inflation_config is None, default behavior is preserved."""
    from api.src.schemas import AuctionConfigSchema

    schema = AuctionConfigSchema(num_participants=8)
    assert schema.inflation_config is None
