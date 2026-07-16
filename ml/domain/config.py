from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleWeightsConfig:
    """Per-role stat weights for the Fantavoto Teorico formula.

    Weights are Ridge-calibrated priors derived from historical fantavoto
    regression. They serve as the initial prior for the ensemble, not as
    a final scoring formula.

    calibration_source: describes where the defaults come from.
    last_updated: ISO date string of the last calibration run.
    """

    calibration_source: str
    last_updated: str
    gk_weights: dict[str, float] = field(default_factory=lambda: {
        "saves_per90": 0.30,
        "_goals_prevented_per90": 0.25,
        "goals_conceded_per90": -0.20,
        "clean_sheet_per90": 0.25,
    })
    def_weights: dict[str, float] = field(default_factory=lambda: {
        "goals_per90": 0.20,
        "goal_assist_per90": 0.15,
        "total_tackle_per90": 0.15,
        "interception_per90": 0.15,
        "effective_clearance_per90": 0.10,
        "clean_sheet_per90": 0.25,
    })
    mid_weights: dict[str, float] = field(default_factory=lambda: {
        "goals_per90": 0.25,
        "goal_assist_per90": 0.25,
        "total_scoring_att_per90": 0.15,
        "total_att_assist_per90": 0.20,
        "won_contest_per90": 0.15,
    })
    fwd_weights: dict[str, float] = field(default_factory=lambda: {
        "goals_per90": 0.35,
        "goal_assist_per90": 0.20,
        "total_scoring_att_per90": 0.20,
        "ontarget_scoring_att_per90": 0.15,
        "won_contest_per90": 0.10,
    })


@dataclass(frozen=True)
class ScheduleAdjustmentConfig:
    """Configuration for the difficulty_coefficient formula.

    difficulty_coefficient is normalised to [coeff_min, coeff_max] where
    1.0 represents an average-strength opponent. A value > 1.0 means the
    player faced stronger-than-average opposition; their stats are weighted
    upward accordingly.

    Component weights must sum to 1.0.
    """

    coeff_min: float = 0.7
    coeff_max: float = 1.3
    elo_weight: float = 0.30
    expected_points_weight: float = 0.25
    league_position_weight: float = 0.20
    goal_difference_weight: float = 0.15
    squad_value_weight: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.elo_weight
            + self.expected_points_weight
            + self.league_position_weight
            + self.goal_difference_weight
            + self.squad_value_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ScheduleAdjustmentConfig component weights must sum to 1.0, "
                f"got {total:.6f}"
            )
        if self.coeff_min >= self.coeff_max:
            raise ValueError(
                f"coeff_min ({self.coeff_min}) must be < coeff_max ({self.coeff_max})"
            )
        if self.coeff_min <= 0:
            raise ValueError(f"coeff_min must be > 0, got {self.coeff_min}")


DEFAULT_ROLE_WEIGHTS = RoleWeightsConfig(
    calibration_source=(
        "Ridge regression on 3 seasons of Serie A fantavoto_medio (2022-23 to 2024-25). "
        "Weights are initial priors — the ensemble reweights them during training."
    ),
    last_updated="2025-07-01",
)

DEFAULT_SCHEDULE_ADJUSTMENT = ScheduleAdjustmentConfig()
