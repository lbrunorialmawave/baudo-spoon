from __future__ import annotations

from dataclasses import dataclass

from ml.domain.predictions import PredictionExplanation
from ml.optimizer.models import ROLE_QUOTAS, Player, Role

PlayerV1 = Player
"""Alias for the current Player schema. Downstream-compatible interface."""


@dataclass(frozen=True)
class PlayerV2:
    """Extended player model with ML enrichments.

    Superset of PlayerV1. Use to_player_v1() to obtain the downstream-
    compatible interface consumed by ml.optimizer and ml.auction.
    """

    player_id: str
    name: str
    role: Role
    real_team: str
    cost: int
    projected_score: float
    reliability_weight: float | None = None
    prediction_explanation: PredictionExplanation | None = None
    expected_minutes: float | None = None
    var_value: float | None = None
    expected_auction_price: float | None = None

    def __post_init__(self) -> None:
        if not self.player_id:
            raise ValueError("PlayerV2.player_id must be non-empty")
        if not self.name:
            raise ValueError("PlayerV2.name must be non-empty")
        if self.role not in ROLE_QUOTAS:
            raise ValueError(
                f"PlayerV2.role must be one of {tuple(ROLE_QUOTAS)}, got {self.role!r}"
            )
        if not self.real_team:
            raise ValueError("PlayerV2.real_team must be non-empty")
        if self.cost < 0:
            raise ValueError(f"PlayerV2.cost must be >= 0, got {self.cost}")
        if self.projected_score < 0:
            raise ValueError(
                f"PlayerV2.projected_score must be >= 0, got {self.projected_score}"
            )
        if self.reliability_weight is not None and self.reliability_weight < 0:
            raise ValueError(
                f"PlayerV2.reliability_weight must be >= 0 if provided, "
                f"got {self.reliability_weight}"
            )
        if self.expected_minutes is not None and self.expected_minutes < 0:
            raise ValueError(
                f"PlayerV2.expected_minutes must be >= 0 if provided, "
                f"got {self.expected_minutes}"
            )
        if self.expected_auction_price is not None and self.expected_auction_price < 0:
            raise ValueError(
                f"PlayerV2.expected_auction_price must be >= 0 if provided, "
                f"got {self.expected_auction_price}"
            )


def to_player_v1(p: PlayerV2) -> PlayerV1:
    """Adapter: strips V2-only fields, returns PlayerV1 for downstream compatibility.

    This is the authorised boundary between the new ML system and the
    downstream optimizer and auction tracker modules. Always use this
    function — never manually reconstruct a Player from PlayerV2 fields.
    """
    return PlayerV1(
        player_id=p.player_id,
        name=p.name,
        role=p.role,
        real_team=p.real_team,
        cost=p.cost,
        projected_score=p.projected_score,
        reliability_weight=p.reliability_weight,
    )
