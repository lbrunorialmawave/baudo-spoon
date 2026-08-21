"""Trade Fairness Engine — bilateral exchange evaluation (Classic & Mantra).

Builds on ``TradePlayer`` / coverage helpers already present in advisor.py.
Does **not** write to the roster store: pure simulation + scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from ml.optimizer.formations import evaluate_coverage, get_formation
from ml.trades.advisor import TradePlayer, _coverage_player_view

log = logging.getLogger(__name__)

Mode = Literal["classic", "mantra"]
Verdict = Literal["vantaggioso", "equilibrato", "sfavorevole"]
Confidence = Literal["assente", "bassa", "media", "alta"]

# Classic role buckets used for validation + per-role fairness
CLASSIC_ROLES = ("GK", "DEF", "MID", "FWD")

# Mantra role → classic bucket (for optional reporting only)
_MANTRA_TO_CLASSIC: dict[str, str] = {
    "Por": "GK",
    "Dd": "DEF", "Ds": "DEF", "Dc": "DEF", "B": "DEF",
    "E": "MID", "M": "MID", "C": "MID", "W": "MID", "T": "MID",
    "A": "FWD", "Pc": "FWD",
}


@dataclass(frozen=True, slots=True)
class PTVWeights:
    base: float = 0.55
    forma: float = 0.25
    titolarita: float = 0.20

    def __post_init__(self) -> None:
        s = self.base + self.forma + self.titolarita
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"PTV weights must sum to 1.0, got {s}")


DEFAULT_WEIGHTS = PTVWeights()


@dataclass(slots=True)
class EnrichedTradePlayer:
    """TradePlayer + live signals needed for PTV."""

    player: TradePlayer
    forma_recente: Optional[float] = None          # 0-100 or None
    games_available_for_form: int = 0              # 0..5+
    indice_titolarita: float = 50.0                # 0-100
    status: str = "unknown"                        # starter/bench/injured/...
    classic_role: str = "MID"                      # GK/DEF/MID/FWD


@dataclass(frozen=True, slots=True)
class PTVResult:
    score: float
    confidence: Confidence
    flags: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlayerPTVView:
    player_id: str
    name: str
    ptv: float
    confidence: Confidence
    flags: list[str]
    classic_role: str
    breakdown: dict[str, float]


@dataclass(frozen=True, slots=True)
class SquadImpact:
    coverage_before: dict[str, bool]
    coverage_after: dict[str, bool]
    warning: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TradeEvaluation:
    mode: Mode
    valid: bool
    validation_errors: list[str]
    verdict: Optional[Verdict]
    value_delta_percent: Optional[float]
    tolerance_band_percent: float
    give: list[PlayerPTVView]
    receive: list[PlayerPTVView]
    squad_impact: Optional[SquadImpact]
    rationale: list[str]
    season_notice: Optional[str] = None


# ── PTV core ─────────────────────────────────────────────────────────────────


def player_trade_value(
    p: EnrichedTradePlayer,
    *,
    weights: PTVWeights = DEFAULT_WEIGHTS,
) -> PTVResult:
    """Combine structural value, recent form and titolarità into a single score.

    Form weight is ramped by ``games_available_for_form / 5`` so cold-start
    and long-term absences never inject a silent default of 50.
    """
    base = float(p.player.fp_corr)
    forma = p.forma_recente
    titolarita = float(p.indice_titolarita)
    games = max(0, int(p.games_available_for_form))

    ramp = min(games / 5.0, 1.0) if forma is not None else 0.0
    peso_forma = weights.forma * ramp
    peso_base = weights.base + weights.forma * (1.0 - ramp)
    peso_tit = weights.titolarita

    if forma is None or games == 0:
        score = base * peso_base + titolarita * peso_tit
        # renormalise because form weight was dropped
        denom = peso_base + peso_tit
        if denom > 0:
            score = (base * peso_base + titolarita * peso_tit) / denom * (
                weights.base + weights.forma + weights.titolarita
            )
        confidence: Confidence = "assente" if forma is None else "bassa"
        forma_used = None
    else:
        score = base * peso_base + float(forma) * peso_forma + titolarita * peso_tit
        if games <= 2:
            confidence = "bassa"
        elif games <= 4:
            confidence = "media"
        else:
            confidence = "alta"
        forma_used = float(forma)

    flags: list[str] = []
    if p.status in ("injured", "suspended"):
        flags.append(f"Indisponibile ({p.status})")
    if titolarita < 40:
        flags.append("Rischio panchina")

    return PTVResult(
        score=round(score, 2),
        confidence=confidence,
        flags=flags,
        breakdown={
            "base": round(base, 2),
            "forma": round(forma_used, 2) if forma_used is not None else None,  # type: ignore[dict-item]
            "titolarita": round(titolarita, 2),
            "peso_base": round(peso_base, 3),
            "peso_forma": round(peso_forma, 3),
            "peso_titolarita": round(peso_tit, 3),
            "games_available": float(games),
        },
    )


# ── Classic validation ───────────────────────────────────────────────────────


def _classic_role_of(p: EnrichedTradePlayer) -> str:
    if p.classic_role in CLASSIC_ROLES:
        return p.classic_role
    # fallback from Mantra roles
    for r in p.player.eligible_roles:
        mapped = _MANTRA_TO_CLASSIC.get(r)
        if mapped:
            return mapped
    return "MID"


def validate_classic(
    give: Sequence[EnrichedTradePlayer],
    receive: Sequence[EnrichedTradePlayer],
) -> list[str]:
    """Same number of pieces per classic role on each side."""
    errors: list[str] = []
    from collections import Counter

    give_roles = Counter(_classic_role_of(p) for p in give)
    recv_roles = Counter(_classic_role_of(p) for p in receive)

    if sum(give_roles.values()) != sum(recv_roles.values()):
        errors.append(
            f"Numero pedine diverso: cedi {sum(give_roles.values())}, "
            f"ricevi {sum(recv_roles.values())}"
        )

    for role in CLASSIC_ROLES:
        g = give_roles.get(role, 0)
        r = recv_roles.get(role, 0)
        if g != r:
            errors.append(
                f"Ruolo {role}: cedi {g}, ricevi {r} "
                "(in Classic ogni ruolo deve bilanciarsi)"
            )
    return errors


# ── Coverage impact (Mantra) ─────────────────────────────────────────────────


def _coverage_map(
    roster: Sequence[TradePlayer],
    formation_prefs: Sequence[str],
) -> dict[str, bool]:
    views = [_coverage_player_view(p) for p in roster]
    result: dict[str, bool] = {}
    for name in formation_prefs:
        try:
            formation = get_formation(name)
            cov = evaluate_coverage(views, formation)
            result[name] = bool(cov.feasible)
        except Exception as exc:  # noqa: BLE001
            log.debug("Coverage eval failed for %s: %s", name, exc)
            result[name] = False
    return result


def compute_squad_impact(
    current_roster: Sequence[TradePlayer],
    give_ids: set[str],
    receive_players: Sequence[TradePlayer],
    formation_prefs: Sequence[str],
) -> SquadImpact:
    before = _coverage_map(current_roster, formation_prefs)
    after_roster = [
        p for p in current_roster if p.player_id not in give_ids
    ] + list(receive_players)
    after = _coverage_map(after_roster, formation_prefs)

    lost = [f for f, ok in before.items() if ok and not after.get(f, False)]
    warning = None
    if lost:
        warning = (
            "Lo scambio elimina la copertura del modulo "
            + (", ".join(lost))
        )
    return SquadImpact(
        coverage_before=before,
        coverage_after=after,
        warning=warning,
    )


# ── Public entry point ───────────────────────────────────────────────────────


def evaluate_trade(
    *,
    mode: Mode,
    give: Sequence[EnrichedTradePlayer],
    receive: Sequence[EnrichedTradePlayer],
    current_roster: Sequence[TradePlayer] | None = None,
    formation_prefs: Sequence[str] = ("4-3-3", "3-5-2", "3-4-3"),
    tolerance_percent: float = 10.0,
    weights: PTVWeights = DEFAULT_WEIGHTS,
    season_notice: str | None = None,
) -> TradeEvaluation:
    """Evaluate a bilateral trade and return a transparent verdict."""

    validation_errors: list[str] = []
    if mode == "classic":
        validation_errors = validate_classic(give, receive)

    valid = len(validation_errors) == 0

    give_ptv = [player_trade_value(p, weights=weights) for p in give]
    recv_ptv = [player_trade_value(p, weights=weights) for p in receive]

    give_views = [
        PlayerPTVView(
            player_id=p.player.player_id,
            name=p.player.name,
            ptv=ptv.score,
            confidence=ptv.confidence,
            flags=list(ptv.flags),
            classic_role=_classic_role_of(p),
            breakdown=dict(ptv.breakdown),
        )
        for p, ptv in zip(give, give_ptv)
    ]
    recv_views = [
        PlayerPTVView(
            player_id=p.player.player_id,
            name=p.player.name,
            ptv=ptv.score,
            confidence=ptv.confidence,
            flags=list(ptv.flags),
            classic_role=_classic_role_of(p),
            breakdown=dict(ptv.breakdown),
        )
        for p, ptv in zip(receive, recv_ptv)
    ]

    sum_give = sum(v.ptv for v in give_views) if give_views else 0.0
    sum_recv = sum(v.ptv for v in recv_views) if recv_views else 0.0

    if sum_give > 0:
        delta_pct = (sum_recv - sum_give) / sum_give * 100.0
    else:
        delta_pct = 0.0 if sum_recv == 0 else 100.0

    if abs(delta_pct) <= tolerance_percent:
        verdict: Optional[Verdict] = "equilibrato"
    elif delta_pct > 0:
        verdict = "vantaggioso"
    else:
        verdict = "sfavorevole"

    if not valid:
        verdict = None

    # Classic: also surface per-role imbalance in rationale even when valid
    rationale: list[str] = []
    if valid:
        rationale.append(
            f"Valore ceduto: {sum_give:.1f} · Valore ricevuto: {sum_recv:.1f} "
            f"({delta_pct:+.1f}%)"
        )
        low_conf = [
            v.name for v in give_views + recv_views if v.confidence in ("assente", "bassa")
        ]
        if low_conf:
            rationale.append(
                f"Attenzione: confidenza bassa/assente su {', '.join(low_conf)}"
            )
        flagged = [
            f"{v.name} ({', '.join(v.flags)})"
            for v in give_views + recv_views
            if v.flags
        ]
        if flagged:
            rationale.append("Flag: " + "; ".join(flagged))
    else:
        rationale.extend(validation_errors)

    squad_impact: Optional[SquadImpact] = None
    if mode == "mantra" and current_roster is not None:
        give_ids = {p.player.player_id for p in give}
        recv_players = [p.player for p in receive]
        squad_impact = compute_squad_impact(
            current_roster, give_ids, recv_players, formation_prefs
        )
        if squad_impact.warning:
            rationale.insert(0, squad_impact.warning)

    return TradeEvaluation(
        mode=mode,
        valid=valid,
        validation_errors=validation_errors,
        verdict=verdict,
        value_delta_percent=round(delta_pct, 2) if valid else None,
        tolerance_band_percent=tolerance_percent,
        give=give_views,
        receive=recv_views,
        squad_impact=squad_impact,
        rationale=rationale,
        season_notice=season_notice,
    )
