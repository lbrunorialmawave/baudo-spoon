"""Domain contracts for the Fantacalcio player evaluation system.

Public re-exports for convenience.
"""

from ml.domain.features import Feature, MissingDataPolicy
from ml.domain.player_versions import PlayerV1, PlayerV2, to_player_v1
from ml.domain.predictions import (
    SHAP_TOLERANCE,
    PredictionExplanation,
    derive_season_value_columns,
    resolve_season_value_fields,
)
from ml.domain.targets import (
    BONUS_PREVISTI,
    FANTAPUNTI_TOTALI,
    FANTAVOTO_MEDIO,
    MINUTI_GIOCATI,
    PREZZO_ATTESO,
    PROBABILITA_TITOLARITA,
    TargetSpec,
)

__all__ = [
    "BONUS_PREVISTI",
    "FANTAPUNTI_TOTALI",
    "FANTAVOTO_MEDIO",
    "MINUTI_GIOCATI",
    "PREZZO_ATTESO",
    "PROBABILITA_TITOLARITA",
    "SHAP_TOLERANCE",
    "Feature",
    "MissingDataPolicy",
    "PlayerV1",
    "PlayerV2",
    "PredictionExplanation",
    "TargetSpec",
    "derive_season_value_columns",
    "resolve_season_value_fields",
    "to_player_v1",
]
