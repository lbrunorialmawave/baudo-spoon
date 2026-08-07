"""TargetBuilder: attaches all 6 target columns to a player-season DataFrame.

Each target is built from the available stats and clipped/validated per its
TargetSpec. All operations are pandas-based for sklearn compatibility.

Target derivation logic:
  - fantavoto_medio: from existing ml.data.target (external CSV or approximation)
  - fantapunti_totali: fantavoto_medio * appearances (season total)
  - bonus_previsti: Σ(goals*3 + assists*1 + clean_sheet_bonus + ...) per season
  - minuti_giocati: mins_played (or appearances * 90 as proxy)
  - probabilita_titolarita: mins_played / (appearances * 90), clipped [0,1]
  - prezzo_atteso: qt_a (Fantacalcio listino) normalised; fallback = role_median
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.domain.targets import (
    BONUS_PREVISTI,
    FANTAPUNTI_TOTALI,
    FANTAVOTO_MEDIO,
    MINUTI_GIOCATI,
    PREZZO_ATTESO,
    PROBABILITA_TITOLARITA,
    TargetSpec,
)

log = logging.getLogger(__name__)

TARGET_SPECS: list[TargetSpec] = [
    FANTAVOTO_MEDIO,
    FANTAPUNTI_TOTALI,
    BONUS_PREVISTI,
    MINUTI_GIOCATI,
    PROBABILITA_TITOLARITA,
    PREZZO_ATTESO,
]

# Fantacalcio bonus values per event (classic rules)
_BONUS_WEIGHTS: dict[str, float] = {
    "goals": 3.0,
    "goal_assist": 1.0,
    "penalty_scored": 3.0,
    "penalty_missed": -3.0,
    "own_goals": -2.0,
    "yellow_card": -0.5,
    "red_card": -1.0,
}
# GK/DEF clean sheet bonus (flat per season, not per-90)
_CLEAN_SHEET_BONUS: dict[str, float] = {"GK": 1.0, "DEF": 1.5}


class TargetBuilder:
    """Derives all 6 target columns from a feature-engineered player-season DataFrame.

    Usage::

        builder = TargetBuilder()
        df_with_targets = builder.build(df_engineered, external_fantavoto_csv=path)

    The builder does NOT fit any model. It is a pure transformation.
    It is stateless: each call to build() is independent.

    Dependencies:
        - fantavoto_medio: produced by ml.data.target.attach_target (or passed in)
        - All other targets: derived from stat columns present in df
        - prezzo_atteso: from qt_a column (Fantacalcio quotation); role-median fallback
    """

    def build(
        self,
        df: pd.DataFrame,
        external_fantavoto_csv: str | None = None,
    ) -> pd.DataFrame:
        """Add all 6 target columns to df. Returns a new copy.

        Columns added: fantavoto_medio, fantapunti_totali, bonus_previsti,
        minuti_giocati, probabilita_titolarita, prezzo_atteso.

        Args:
            df: Feature-engineered DataFrame (output of engineer_features or similar).
                Must contain 'appearances' or 'mins_played' at minimum.
            external_fantavoto_csv: Optional CSV path for actual fantavoto data.

        Returns:
            New DataFrame with all target columns added (existing cols preserved).
        """
        df = df.copy()

        # 1. fantavoto_medio (primary target — attach from CSV or approximate)
        if "fantavoto_medio" not in df.columns or df["fantavoto_medio"].isna().all():
            df = self._attach_fantavoto(df, external_fantavoto_csv)

        # 2. fantapunti_totali = fantavoto_medio * appearances
        df = self._build_fantapunti_totali(df)

        # 3. bonus_previsti = season-total bonus events (weighted sum of counting stats)
        df = self._build_bonus_previsti(df)

        # 4. minuti_giocati
        df = self._build_minuti_giocati(df)

        # 5. probabilita_titolarita
        df = self._build_probabilita_titolarita(df)

        # 6. prezzo_atteso
        df = self._build_prezzo_atteso(df)

        return df

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _attach_fantavoto(
        self, df: pd.DataFrame, external_csv: str | None
    ) -> pd.DataFrame:
        from pathlib import Path

        from ml.data.target import attach_target

        csv_path = Path(external_csv) if external_csv else None
        return attach_target(df, external_csv=csv_path, min_minutes=0)
        # min_minutes=0 here: TargetBuilder does not apply quality filter itself;
        # the Trainer handles that separately.

    def _build_fantapunti_totali(self, df: pd.DataFrame) -> pd.DataFrame:
        """fantapunti_totali = fantavoto_medio * appearances (approx season total)."""
        apps = self._get_appearances(df)
        if "fantavoto_medio" in df.columns:
            df["fantapunti_totali"] = (df["fantavoto_medio"].fillna(6.0) * apps).clip(
                lower=0.0
            )
        else:
            log.warning("fantavoto_medio not available; fantapunti_totali set to NaN.")
            df["fantapunti_totali"] = np.nan
        return df

    def _build_bonus_previsti(self, df: pd.DataFrame) -> pd.DataFrame:
        """bonus_previsti = Σ(event * bonus_weight) for the season."""
        bonus = pd.Series(0.0, index=df.index)
        for col, weight in _BONUS_WEIGHTS.items():
            if col in df.columns:
                bonus += pd.to_numeric(df[col], errors="coerce").fillna(0.0) * weight

        # Clean sheet bonus for GK/DEF (season-level clean sheets, not per-90)
        if "clean_sheet" in df.columns and "canonical_role" in df.columns:
            for role, cs_bonus in _CLEAN_SHEET_BONUS.items():
                mask = df["canonical_role"] == role
                if mask.any():
                    cs = pd.to_numeric(
                        df.loc[mask, "clean_sheet"], errors="coerce"
                    ).fillna(0.0)
                    bonus.loc[mask] += cs * cs_bonus

        df["bonus_previsti"] = bonus.clip(lower=0.0)
        return df

    def _build_minuti_giocati(self, df: pd.DataFrame) -> pd.DataFrame:
        """minuti_giocati from mins_played; fallback: appearances * 90."""
        if "mins_played" in df.columns:
            df["minuti_giocati"] = (
                pd.to_numeric(df["mins_played"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
        elif "appearances" in df.columns:
            log.warning(
                "mins_played absent; estimating minuti_giocati as appearances * 90."
            )
            df["minuti_giocati"] = (
                pd.to_numeric(df["appearances"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                * 90.0
            )
        else:
            log.warning(
                "Neither mins_played nor appearances found; minuti_giocati = NaN."
            )
            df["minuti_giocati"] = np.nan
        return df

    def _build_probabilita_titolarita(self, df: pd.DataFrame) -> pd.DataFrame:
        """probabilita_titolarita = mins_played / (appearances * 90), clipped [0, 1].

        Approximates the fraction of available minutes actually played — a proxy
        for starting probability.
        """
        apps = self._get_appearances(df)
        if "mins_played" in df.columns:
            mins = pd.to_numeric(df["mins_played"], errors="coerce").fillna(0.0)
            max_possible = apps * 90.0
            df["probabilita_titolarita"] = (mins / max_possible.clip(lower=1.0)).clip(
                0.0, 1.0
            )
        else:
            # Without minutes: use appearances / expected_max_appearances (38 for Serie A)
            expected_max = 38.0
            df["probabilita_titolarita"] = (apps.clip(lower=0.0) / expected_max).clip(
                0.0, 1.0
            )
            log.warning(
                "mins_played absent; probabilita_titolarita approximated from appearances / 38."
            )
        return df

    def _build_prezzo_atteso(self, df: pd.DataFrame) -> pd.DataFrame:
        """prezzo_atteso from qt_a (Fantacalcio listino in credits).

        Falls back to role-median when qt_a is unavailable.
        """
        if "qt_a" in df.columns:
            qt = pd.to_numeric(df["qt_a"], errors="coerce")
            # Role-median fallback for NaN values
            if qt.isna().any() and "canonical_role" in df.columns:
                role_medians = qt.groupby(df["canonical_role"]).transform("median")
                qt = qt.fillna(role_medians)
            global_median = qt.median()
            qt = qt.fillna(global_median if not np.isnan(global_median) else 1.0)
            df["prezzo_atteso"] = qt.clip(lower=1.0)
        else:
            log.warning(
                "qt_a column absent; prezzo_atteso set to role-median of 1.0 (uncalibrated). "
                "Run ml.data.import_quotations to populate Fantacalcio quotation data."
            )
            df["prezzo_atteso"] = 1.0
        return df

    @staticmethod
    def _get_appearances(df: pd.DataFrame) -> pd.Series:
        if "appearances" in df.columns:
            return (
                pd.to_numeric(df["appearances"], errors="coerce")
                .fillna(1.0)
                .clip(lower=1.0)
            )
        if "mins_played" in df.columns:
            return (
                pd.to_numeric(df["mins_played"], errors="coerce").fillna(90.0) / 90.0
            ).clip(lower=1.0)
        return pd.Series(30.0, index=df.index)
