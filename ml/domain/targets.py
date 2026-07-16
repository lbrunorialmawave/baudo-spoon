from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import polars as pl


@dataclass(frozen=True)
class TargetSpec:
    """Specification for a prediction target variable.

    Args:
        name: Column name for the target.
        target_type: Nature of the prediction task.
        transform: Optional monotonic transform applied before training.
        inverse_transform: Inverse of `transform`, applied to model outputs.
    """

    name: str
    target_type: Literal["regression", "classification", "probability"]
    transform: Callable[[pl.Series], pl.Series] | None = None
    inverse_transform: Callable[[pl.Series], pl.Series] | None = None


def _log1p(s: pl.Series) -> pl.Series:
    return s.log1p()


def _expm1(s: pl.Series) -> pl.Series:
    return s.exp() - 1.0


FANTAVOTO_MEDIO = TargetSpec(
    name="fantavoto_medio",
    target_type="regression",
)

FANTAPUNTI_TOTALI = TargetSpec(
    name="fantapunti_totali",
    target_type="regression",
    transform=_log1p,
    inverse_transform=_expm1,
)

BONUS_PREVISTI = TargetSpec(
    name="bonus_previsti",
    target_type="regression",
    transform=_log1p,
    inverse_transform=_expm1,
)

MINUTI_GIOCATI = TargetSpec(
    name="minuti_giocati",
    target_type="regression",
)

PROBABILITA_TITOLARITA = TargetSpec(
    name="probabilita_titolarita",
    target_type="probability",
)

PREZZO_ATTESO = TargetSpec(
    name="prezzo_atteso",
    target_type="regression",
    transform=_log1p,
    inverse_transform=_expm1,
)
