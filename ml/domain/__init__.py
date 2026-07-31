"""Domain contracts for the Fantacalcio player evaluation system.

Public re-exports for convenience.
"""

from ml.domain.features import Feature, MissingDataPolicy
from ml.domain.predictions import (
    PredictionExplanation,
    SHAP_TOLERANCE,
    derive_season_value_columns,
    resolve_season_value_fields,
)
from ml.domain.player_versions import PlayerV1, PlayerV2, to_player_v1
from ml.domain.targets import (
    TargetSpec,
    FANTAVOTO_MEDIO,
    FANTAPUNTI_TOTALI,
    BONUS_PREVISTI,
    MINUTI_GIOCATI,
    PROBABILITA_TITOLARITA,
    PREZZO_ATTESO,
)

__all__ = [
    "Feature",
    "MissingDataPolicy",
    "PredictionExplanation",
    "SHAP_TOLERANCE",
    "derive_season_value_columns",
    "resolve_season_value_fields",
    "PlayerV1",
    "PlayerV2",
    "to_player_v1",
    "TargetSpec",
    "FANTAVOTO_MEDIO",
    "FANTAPUNTI_TOTALI",
    "BONUS_PREVISTI",
    "MINUTI_GIOCATI",
    "PROBABILITA_TITOLARITA",
    "PREZZO_ATTESO",
]
