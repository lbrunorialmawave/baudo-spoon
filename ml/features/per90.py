"""Per-90 normalised stat features.

Each concrete subclass wraps exactly one stat column.  The denominator is
``mins_played / 90`` (clipped at 1.0) when available, falling back to
``appearances`` for older data that lacks granular minutes.

All subclasses use ``MissingDataPolicy.IMPUTE_ZERO``: missing stat columns
are filled with 0 before dividing, matching the existing pandas behaviour in
``ml/preprocessing/features.py:add_per90_features()``.
"""

from __future__ import annotations

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = ["ALL_PER90_FEATURES", "Per90Feature"]


class Per90Feature(Feature):
    """Base per-90 feature.  Subclasses set ``stat_col`` at class level.

    ``name`` and ``required_columns`` are derived from ``stat_col`` so each
    subclass only needs a one-liner body.
    """

    stat_col: str
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.stat_col}_per90"

    @property
    def required_columns(self) -> frozenset[str]:  # type: ignore[override]
        return frozenset([self.stat_col])

    def compute(self, data: pl.DataFrame) -> pl.Series:
        if "mins_played" in data.columns:
            denom = (data["mins_played"].cast(pl.Float64) / 90.0).clip(lower_bound=1.0)
        elif "appearances" in data.columns:
            denom = data["appearances"].cast(pl.Float64).clip(lower_bound=1.0)
        else:
            raise ValueError(
                "Per90Feature requires 'mins_played' or 'appearances' column."
            )
        stat = data[self.stat_col].cast(pl.Float64).fill_null(0.0)
        return stat / denom


# ── Concrete subclasses (one per stat in _PER_90_CANDIDATES) ──────────────────


class GoalsPer90(Per90Feature):
    stat_col = "goals"


class GoalAssistPer90(Per90Feature):
    stat_col = "goal_assist"


class TotalScoringAttPer90(Per90Feature):
    stat_col = "total_scoring_att"


class OntargetScoringAttPer90(Per90Feature):
    stat_col = "ontarget_scoring_att"


class BigChanceCreatedPer90(Per90Feature):
    stat_col = "big_chance_created"


class BigChanceMissedPer90(Per90Feature):
    stat_col = "big_chance_missed"


class TotalAttAssistPer90(Per90Feature):
    stat_col = "total_att_assist"


class WonContestPer90(Per90Feature):
    stat_col = "won_contest"


class YellowCardPer90(Per90Feature):
    stat_col = "yellow_card"


class RedCardPer90(Per90Feature):
    stat_col = "red_card"


class PenaltyWonPer90(Per90Feature):
    stat_col = "penalty_won"


class PenaltyConcededPer90(Per90Feature):
    stat_col = "penalty_conceded"


class OutfielderBlockPer90(Per90Feature):
    stat_col = "outfielder_block"


class InterceptionPer90(Per90Feature):
    stat_col = "interception"


class TotalTacklePer90(Per90Feature):
    stat_col = "total_tackle"


class EffectiveClearancePer90(Per90Feature):
    stat_col = "effective_clearance"


class AccuratePassPer90(Per90Feature):
    stat_col = "accurate_pass"


class FoulsPer90(Per90Feature):
    stat_col = "fouls"


class SavesPer90(Per90Feature):
    stat_col = "saves"
    applicable_roles: frozenset[str] = frozenset(["GK"])


class GoalsPreventedPer90(Per90Feature):
    stat_col = "_goals_prevented"
    applicable_roles: frozenset[str] = frozenset(["GK"])


class GoalsConcededPer90(Per90Feature):
    stat_col = "goals_conceded"
    applicable_roles: frozenset[str] = frozenset(["GK"])


class CleanSheetPer90(Per90Feature):
    stat_col = "clean_sheet"
    applicable_roles: frozenset[str] = frozenset(["GK", "DEF"])


# ── Module-level instances ────────────────────────────────────────────────────

ALL_PER90_FEATURES: list[Per90Feature] = [
    GoalsPer90(),
    GoalAssistPer90(),
    TotalScoringAttPer90(),
    OntargetScoringAttPer90(),
    BigChanceCreatedPer90(),
    BigChanceMissedPer90(),
    TotalAttAssistPer90(),
    WonContestPer90(),
    YellowCardPer90(),
    RedCardPer90(),
    PenaltyWonPer90(),
    PenaltyConcededPer90(),
    OutfielderBlockPer90(),
    InterceptionPer90(),
    TotalTacklePer90(),
    EffectiveClearancePer90(),
    AccuratePassPer90(),
    FoulsPer90(),
    SavesPer90(),
    GoalsPreventedPer90(),
    GoalsConcededPer90(),
    CleanSheetPer90(),
]
