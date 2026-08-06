"""P3 — Peso Squadra (team strength context pillar).

PS_corretto is computed as a weighted average of 5 normalised parameters:

    PS_corretto = team_rank_norm_pct × 0.27
                + prev_season_points_pct × 0.22
                + goal_difference_pct × 0.17
                + avg_team_rating_pct × 0.17
                + squad_value_market_pct × 0.17

Then:

    Moltiplicatore = 1 + max(0, (PS_corretto - 50) * Coeff_Base)
    Max_Moltiplicatore = 1 + (100 - 50) * Coeff_Base
    P3 = clip(PS_corretto * Moltiplicatore / Max_Moltiplicatore, 0, 100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.mantra.config import MantraConfig


def _normalise_pct(series: pd.Series) -> pd.Series:
    """Min-max normalise *series* to [0, 100].

    Returns 50 for all values when the range is zero.
    """
    s = series.fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(50.0, index=s.index)
    return ((s - s_min) / (s_max - s_min)) * 100.0


def compute_ps_corretto(df: pd.DataFrame, cfg: MantraConfig) -> pd.Series:
    """Compute PS_corretto (0-100) for each team from multiple parameters.

    Parameters
    ----------
    df:
        DataFrame with one row per **team-season** (not per player), with:
        - ``team_rank_norm``       — FotMob team rank (0-1, 1=best)
        - ``prev_season_points``   — points in previous season
        - ``goal_difference``      — season goal difference
        - ``avg_team_rating``      — mean FotMob rating of players in team
        - ``squad_value_market``   — SUM(qt_a) for the team's players
        - ``season_start``         — season identifier
    cfg:
        Calibrated coefficients with PS weight parameters.

    Returns
    -------
    pd.Series of PS_corretto values per row.
    """
    components = {}

    # Each parameter normalised 0-100 within season
    for col, param_name in [
        ("team_rank_norm", "team_rank"),
        ("prev_season_points", "prev_points"),
        ("goal_difference", "goal_diff"),
        ("avg_team_rating", "avg_rating"),
        ("squad_value_market", "squad_value"),
    ]:
        if col in df.columns and df[col].notna().any():
            # Normalise within each season
            if "season_start" in df.columns:
                normed = df.groupby("season_start", group_keys=False)[col].transform(
                    _normalise_pct
                )
            else:
                normed = _normalise_pct(df[col])
            components[param_name] = normed
        else:
            components[param_name] = pd.Series(50.0, index=df.index)

    # Weighted average
    ps = (
        components["team_rank"]    * cfg.PS_TEAM_RANK_WEIGHT
        + components["prev_points"] * cfg.PS_PREV_POINTS_WEIGHT
        + components["goal_diff"]   * cfg.PS_GOAL_DIFF_WEIGHT
        + components["avg_rating"]  * cfg.PS_AVG_RATING_WEIGHT
        + components["squad_value"] * cfg.PS_SQUAD_VALUE_WEIGHT
    )
    return ps.clip(lower=0, upper=100)


def compute_p3(
    player_df: pd.DataFrame,
    team_ps: pd.Series,
    cfg: MantraConfig,
) -> pd.Series:
    """Compute P3 for each player using pre-computed team PS_corretto.

    Parameters
    ----------
    player_df:
        Player-level DataFrame with ``ruolo_primario`` column.
    team_ps:
        Series (indexed by team or merged) with PS_corretto per team-season.
    cfg:
        MantraConfig with COEFF_BASE per role.

    Returns
    -------
    pd.Series of P3 values clipped to [0, 100].
    """
    ps = team_ps.fillna(50).clip(0, 100).astype(float)

    # Get role-specific coefficient
    coeff = player_df["ruolo_primario"].map(cfg.COEFF_BASE).fillna(0.003)

    moltiplicatore = 1.0 + np.maximum(0.0, (ps - 50.0) * coeff)
    max_moltiplicatore = 1.0 + (100.0 - 50.0) * coeff

    p3 = ps * moltiplicatore / max_moltiplicatore
    return p3.clip(lower=0, upper=100)
