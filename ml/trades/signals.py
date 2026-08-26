"""Live signals for the Trade Fairness Engine: Forma_Recente + Indice_Titolarita.

Pure computation helpers. DB access lives in the API layer (or a thin
repository helper) so this module stays unit-testable without Postgres.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence


# ── Forma recente (EWMA of matchday fantavoto) ───────────────────────────────

DEFAULT_EWMA_LAMBDA = 0.65
DEFAULT_FORM_WINDOW = 5


@dataclass(frozen=True, slots=True)
class MatchdayVote:
    """Minimal vote row used by the EWMA calculator."""

    giornata: int
    fantavoto: Optional[float]  # None = s.v. / did not play


@dataclass(frozen=True, slots=True)
class FormaResult:
    forma: Optional[float]  # 0-100 scale, None if no usable votes
    games_available: int
    ewma_raw: Optional[float]  # raw fantavoto EWMA (≈4-10 scale)
    confidence: str  # assente | bassa | media | alta


def ewma_fantavoto(
    votes: Sequence[MatchdayVote],
    *,
    lam: float = DEFAULT_EWMA_LAMBDA,
    window: int = DEFAULT_FORM_WINDOW,
) -> tuple[Optional[float], int]:
    """Exponentially weighted moving average of recent fantavoto.

    Only rows with a numeric fantavoto count. Most recent giornata first
    in the decay (newest gets weight 1, previous λ, then λ², …).
    Returns (ewma, games_used).
    """
    usable = sorted(
        (v for v in votes if v.fantavoto is not None),
        key=lambda v: v.giornata,
        reverse=True,
    )[:window]
    if not usable:
        return None, 0

    num = 0.0
    den = 0.0
    w = 1.0
    for v in usable:
        num += w * float(v.fantavoto)  # type: ignore[arg-type]
        den += w
        w *= lam
    return num / den, len(usable)


def forma_recente_score(
    votes: Sequence[MatchdayVote],
    *,
    pool_mean: float = 6.0,
    pool_std: float = 0.8,
    lam: float = DEFAULT_EWMA_LAMBDA,
    window: int = DEFAULT_FORM_WINDOW,
) -> FormaResult:
    """Map EWMA fantavoto onto the 0-100 FP_Corr scale via z-score.

    Same statistical principle used by the MANTRA algorithm (clip around
    a role-pool mean). When pool_std is degenerate we fall back to a
    linear map 4→0, 10→100.
    """
    ewma, games = ewma_fantavoto(votes, lam=lam, window=window)
    if ewma is None or games == 0:
        return FormaResult(forma=None, games_available=0, ewma_raw=None, confidence="assente")

    if pool_std and pool_std > 1e-6:
        z = (ewma - pool_mean) / pool_std
        score = 50.0 + z * 15.0
    else:
        # linear fallback: 4 → 0, 10 → 100
        score = (ewma - 4.0) / 6.0 * 100.0

    score = max(0.0, min(100.0, score))

    if games <= 2:
        conf = "bassa"
    elif games <= 4:
        conf = "media"
    else:
        conf = "alta"

    return FormaResult(
        forma=round(score, 2),
        games_available=games,
        ewma_raw=round(ewma, 3),
        confidence=conf,
    )


def pool_stats(
    all_ewmas: Sequence[float],
) -> tuple[float, float]:
    """Mean/std of a role pool of raw EWMA fantavoto values."""
    vals = [v for v in all_ewmas if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return 6.0, 0.8
    return statistics.fmean(vals), statistics.pstdev(vals) or 0.8


# ── Fase7 titolarità attesa (EWMA of matchday-status probability) ───────────


@dataclass(frozen=True, slots=True)
class MatchdayStatusRow:
    """One ``player_matchday_status`` row used by the Fase7 CERTEZZA gate."""

    giornata: int
    probability: float  # 0-100, estimated starting-XI probability
    status: str  # starter | bench | injured | suspended | doubtful | unknown


def ewma_titolarita(
    rows: Sequence[MatchdayStatusRow],
    *,
    lam: float = DEFAULT_EWMA_LAMBDA,
    window: int = DEFAULT_FORM_WINDOW,
) -> tuple[Optional[float], int]:
    """Exponentially weighted moving average of recent starting-XI probability.

    Same decay as ``ewma_fantavoto`` (most recent giornata weight 1, then
    lam, lam², …). ``injured``/``suspended`` rows count as probability 0
    regardless of the stored value: a player out injured this week has zero
    real chance of starting, and an ongoing injury should pull the average
    down rather than being skipped. ``probability`` is itself an external
    estimate (not an observed outcome), so this EWMA smooths an estimate —
    acceptable, but callers must not feed it stale rows from a season in
    which the player was on a different team (see the guard in
    ``ml/mantra/runner.py::_compute_titolarita_attesa``).
    """
    out_of_squad = {"injured", "suspended"}
    usable = sorted(rows, key=lambda r: r.giornata, reverse=True)[:window]
    if not usable:
        return None, 0

    num = 0.0
    den = 0.0
    w = 1.0
    for r in usable:
        p = 0.0 if r.status in out_of_squad else float(r.probability)
        num += w * p
        den += w
        w *= lam
    return num / den, len(usable)


# ── Indice titolarità ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TitolaritaInputs:
    """Raw signals available for a player."""

    probability_matchday: Optional[float] = None  # 0-100 from player_matchday_status
    titolarita_esperti: Optional[float] = None  # 1-10 from expert_ratings
    status: str = "unknown"


@dataclass(frozen=True, slots=True)
class TitolaritaResult:
    indice: float  # 0-100
    status: str
    flags: list[str]


def indice_titolarita(
    inputs: TitolaritaInputs,
    *,
    weight_prob: float = 0.6,
    weight_experts: float = 0.4,
) -> TitolaritaResult:
    """Combine current-matchday probability with long-horizon expert titolarità.

    Falls back to whichever signal is present. Defaults to 50 when both missing.
    """
    parts: list[tuple[float, float]] = []  # (value_0_100, weight)

    if inputs.probability_matchday is not None:
        parts.append((max(0.0, min(100.0, float(inputs.probability_matchday))), weight_prob))

    if inputs.titolarita_esperti is not None:
        # 1-10 → 0-100
        scaled = max(0.0, min(100.0, float(inputs.titolarita_esperti) * 10.0))
        parts.append((scaled, weight_experts))

    if not parts:
        indice = 50.0
    else:
        total_w = sum(w for _, w in parts)
        indice = sum(v * w for v, w in parts) / total_w

    flags: list[str] = []
    status = inputs.status or "unknown"
    if status in ("injured", "suspended"):
        flags.append(f"Indisponibile ({status})")
    if indice < 40:
        flags.append("Rischio panchina")

    return TitolaritaResult(indice=round(indice, 2), status=status, flags=flags)
