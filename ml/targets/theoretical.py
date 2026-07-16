"""Theoretical Fantavoto feature.

Computes a role-weighted sum of normalised per-90 stats as a prior estimate
of player quality. This is a feature fed into the ensemble model, not a
final score. The ensemble learns the optimal weight to assign it relative to
historical trend and other features.

Formula:
    theoretical_score(player, role) = BASE_RATING +
        Σ_k (role_weight[role][k] * stat_k_per90)

The weights in RoleWeightsConfig are Ridge-calibrated priors. Using them as
a feature (rather than the output) lets the ensemble correct systematic biases
in the formula automatically.
"""
from __future__ import annotations
import logging
import polars as pl
from ml.domain.features import Feature, MissingDataPolicy
from ml.domain.config import RoleWeightsConfig, DEFAULT_ROLE_WEIGHTS

log = logging.getLogger(__name__)

_BASE_RATING: float = 6.0

# Map canonical_role values to weight dict keys in RoleWeightsConfig
_ROLE_MAP: dict[str, str] = {
    "GK":  "gk_weights",
    "DEF": "def_weights",
    "MID": "mid_weights",
    "FWD": "fwd_weights",
}


class TheoreticalFantavoto(Feature):
    """Role-weighted per-90 score as a prior feature.

    Required columns: canonical_role + all stat columns referenced by weights.
    Missing stat columns: filled with 0.0 (IMPUTE_ZERO).
    Missing canonical_role: FWD weights applied as default.

    Args:
        config: RoleWeightsConfig instance. Defaults to DEFAULT_ROLE_WEIGHTS.
    """

    name = "theoretical_fantavoto"
    required_columns = frozenset(["canonical_role"])
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    def __init__(self, config: RoleWeightsConfig = DEFAULT_ROLE_WEIGHTS) -> None:
        self.config = config

    def compute(self, data: pl.DataFrame) -> pl.Series:
        # ponytail: 4 role × ~5 stat iterations; Polars expr rewrite if called in tight loop
        scores = pl.Series("theoretical_fantavoto", [_BASE_RATING] * len(data))

        role_col = (
            data["canonical_role"].cast(pl.Utf8)
            if "canonical_role" in data.columns
            else pl.Series("canonical_role", ["FWD"] * len(data))
        )

        for role_code, weights_attr in _ROLE_MAP.items():
            weights: dict[str, float] = getattr(self.config, weights_attr)
            role_mask = role_col == role_code
            n_role = role_mask.sum()
            if n_role == 0:
                continue

            role_data = data.filter(role_mask)
            contribution = pl.Series("_contrib", [0.0] * n_role)

            for stat_col, weight in weights.items():
                if stat_col in role_data.columns:
                    vals = role_data[stat_col].cast(pl.Float64).fill_null(0.0)
                else:
                    vals = pl.Series("_zero", [0.0] * n_role)
                contribution = contribution + vals * weight

            # Build full-length series for this role, inserting contributions at role positions
            role_indices = role_mask.arg_true().to_list()
            scores_list = scores.to_list()
            contrib_list = contribution.to_list()
            for i, idx in enumerate(role_indices):
                scores_list[idx] += contrib_list[i]
            scores = pl.Series("theoretical_fantavoto", scores_list)

        # Handle unrecognised roles: apply FWD weights (already defaulted above)
        unknown_mask = ~role_col.is_in(list(_ROLE_MAP.keys()))
        if unknown_mask.sum() > 0:
            log.warning(
                "TheoreticalFantavoto: %d rows with unrecognised canonical_role; FWD weights applied.",
                unknown_mask.sum(),
            )
            fwd_weights = self.config.fwd_weights
            unk_data = data.filter(unknown_mask)
            contribution = pl.Series("_contrib", [0.0] * unknown_mask.sum())
            for stat_col, weight in fwd_weights.items():
                if stat_col in unk_data.columns:
                    vals = unk_data[stat_col].cast(pl.Float64).fill_null(0.0)
                else:
                    vals = pl.Series("_zero", [0.0] * unknown_mask.sum())
                contribution = contribution + vals * weight
            unk_indices = unknown_mask.arg_true().to_list()
            scores_list = scores.to_list()
            contrib_list = contribution.to_list()
            for i, idx in enumerate(unk_indices):
                scores_list[idx] += contrib_list[i]
            scores = pl.Series("theoretical_fantavoto", scores_list)

        return scores.clip(1.0, 10.0)
