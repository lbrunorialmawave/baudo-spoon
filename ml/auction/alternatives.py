"""Pure function: alternative suggestions for a target player.

Given a target ``Player`` and the current ``AuctionState``, the function
returns two candidates restricted to a **compatible role set**:

* ``low_cost_alternative`` - the available player with the best ratio
  ``projected_score / expected_price`` among those whose ``expected_price``
  falls under the configured percentile threshold of the role.
* ``closest_alternative`` - the available player with ``projected_score``
  closest (absolute distance) to the target; ties broken by lower
  ``expected_price``.

Role compatibility
------------------
CLASSIC: candidates must share the same scalar ``role`` as the target
(``p.role == target.role``) — unchanged behaviour.

MANTRA: candidates must have a non-empty intersection between their
``eligible_roles`` and the target's role set (``eligible_roles`` if
present, else ``{role}``). This reuses the same "one player fills at
most one slot" conceptual framing of the ILP solver: a multi-role
player is a valid alternative for any vacant slot they can still cover.

If no compatible candidates remain, the function returns explicit
``None`` results with a diagnostic ``reason_if_none`` (per spec §5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ml.auction.models import (
    AlternativeSuggestion,
    AlternativesConfig,
    AuctionState,
    ValuationMode,
)
from ml.auction.price_drift import project_price_for_player
from ml.optimizer.models import Player

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "suggest_alternatives",
    "player_role_set",
    "pareto_diversify",
    "max_affordable_bid",
    "strategy_price_cap",
]


def player_role_set(player: Player, ruleset: str = "CLASSIC") -> frozenset[str]:
    """Role codes a player can cover under the active ruleset.

    CLASSIC (or missing ``eligible_roles``): singleton ``{player.role}``.
    MANTRA with ``eligible_roles``: the eligible set as-is.
    """
    if ruleset == "MANTRA" and player.eligible_roles:
        return frozenset(player.eligible_roles)
    return frozenset({player.role})


def _roles_compatible(a: Player, b: Player, ruleset: str) -> bool:
    """True iff the two players share at least one coverable role slot."""
    return bool(player_role_set(a, ruleset) & player_role_set(b, ruleset))


def _get_player_score(
    player: Player,
    valuation_mode: ValuationMode,
    *,
    apply_reliability_weight: bool = True,
    risk_aversion: float = 0.0,
) -> float:
    """Decision score for ranking/alternatives (canonical policy).

    Uses :func:`ml.auction.decision_score.compute_decision_score_from_player`
    so alternatives ranking cannot bypass the reliability policy.
    """
    from ml.auction.decision_score import compute_decision_score_from_player

    use_season = valuation_mode == ValuationMode.SEASON_VALUE
    return compute_decision_score_from_player(
        player,
        apply_reliability_weight=apply_reliability_weight,
        risk_aversion=risk_aversion,
        use_season_value=use_season,
    )


def pareto_diversify(
    candidates: list[Player],
    expected_prices: dict[str, float],
    valuation_mode: ValuationMode = ValuationMode.PER_MATCH_RATING,
    *,
    max_points: int = 5,
    exclude_ids: set[str] | None = None,
    apply_reliability_weight: bool = True,
    risk_aversion: float = 0.0,
) -> list[Player]:
    """Mini-fronte Pareto su (score ↑, -price ↑, value_ratio ↑).

    WS3 #3: lightweight adaptation of :mod:`ml.optimizer.pareto` /
    :mod:`ml.optimizer.diversity` for the live alternatives flow — no ILP,
    pure dominance filter over the already-filtered role-compatible pool.

    A candidate A dominates B if it is strictly better on at least one
    axis and not worse on any other. Axes:

    * score = projected (or season) score — higher is better
    * -expected_price — lower price is better
    * value_ratio = score / expected_price — higher is better
    """
    exclude = exclude_ids or set()
    scored: list[tuple[Player, float, float, float]] = []
    for p in candidates:
        if p.player_id in exclude:
            continue
        score = _get_player_score(
            p, valuation_mode,
            apply_reliability_weight=apply_reliability_weight,
            risk_aversion=risk_aversion,
        )
        price = max(1e-9, expected_prices.get(p.player_id, p.cost or 1.0))
        ratio = score / price
        scored.append((p, score, -price, ratio))

    frontier: list[tuple[Player, float, float, float]] = []
    for cand in scored:
        dominated = False
        for other in scored:
            if other[0].player_id == cand[0].player_id:
                continue
            # other dominates cand?
            if (
                other[1] >= cand[1]
                and other[2] >= cand[2]
                and other[3] >= cand[3]
                and (other[1] > cand[1] or other[2] > cand[2] or other[3] > cand[3])
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)

    # Stable order: higher value ratio first, then higher score.
    frontier.sort(key=lambda t: (t[3], t[1]), reverse=True)
    return [t[0] for t in frontier[:max_points]]


def max_affordable_bid(
    state: AuctionState,
    participant_id: str,
) -> int | None:
    """Max bid the participant can place without violating credit reserve.

    WS3 #4 (sensitivity): ``budget_residual - (slots_remaining - 1)``,
    clamped to >= 0. Returns ``None`` if the participant is unknown.
    """
    winner = state.participants.get(participant_id)
    if winner is None:
        return None
    total = sum(int(q) for q in state.config.role_quotas.values())
    current = sum(winner.role_breakdown.values())
    slots_remaining = total - current
    if slots_remaining <= 0:
        return 0
    return max(0, winner.budget_residual - (slots_remaining - 1))


def strategy_price_cap(
    player: Player,
    base_expected_price: float,
    strategy_name: str | None = None,
) -> int | None:
    """Strategy-aware max price threshold (WS3 #5).

    Multiplies the expected price by a role weight from the default
    strategy profiles. Returns ``None`` when no strategy is selected.
    """
    if not strategy_name:
        return None
    try:
        from ml.optimizer.strategies import strategy_by_name
        from ml.optimizer.models import StrategyName

        profile = strategy_by_name(strategy_name)  # type: ignore[arg-type]
    except (KeyError, ImportError, TypeError):
        return None
    weight = profile.role_weight.get(player.role, 1.0)
    # Cap = expected * weight, floored at 1. Higher weight (e.g. D in
    # SUPER_DEFENSIVE) allows paying more for that role.
    return max(1, int(round(base_expected_price * weight)))


def suggest_alternatives(
    target: Player,
    available_pool: list[Player],
    state: AuctionState,
    config: AlternativesConfig,
    valuation_mode: ValuationMode = ValuationMode.PER_MATCH_RATING,
    *,
    participant_id: str | None = None,
    strategy_name: str | None = None,
    diversify: bool = True,
    max_diversified: int = 5,
) -> AlternativeSuggestion:
    """Suggerisce alternative compatibili col ruolo del ``target``.

    Parameters
    ----------
    target:
        Giocatore bersaglio (può essere già stato assegnato — la firma
        ammette entrambi i casi).
    available_pool:
        Pool di giocatori *non ancora assegnati*.  Il filtro di
        compatibilità di ruolo è applicato internamente (CLASSIC:
        uguaglianza scalare; MANTRA: intersezione su ``eligible_roles``).
    state:
        Stato corrente dell'asta, usato per calcolare gli ``expected_price``
        aggiornati e per leggere il ``ruleset``.
    config:
        Configurazione delle soglie (soprattutto ``low_cost_percentile``).
    participant_id:
        Optional: when set, computes ``max_affordable_bid`` for this manager.
    strategy_name:
        Optional: one of BALANCED / SUPER_DEFENSIVE / SUPER_OFFENSIVE / MIXED;
        enables ``strategy_price_cap``.
    diversify:
        When True (default), populate ``diversified_alternatives`` via
        Pareto filter (WS3 #3).

    Returns
    -------
    :class:`AlternativeSuggestion` con i due candidati classici più i
    campi WS3 opzionali.
    """
    ruleset = getattr(state.config, "ruleset", "CLASSIC") or "CLASSIC"
    # Canonical decision policy from auction config (WS6)
    apply_rw = bool(getattr(state.config, "apply_reliability_weight", True))
    risk_av = float(getattr(state.config, "risk_aversion", 0.0) or 0.0)
    same_role = [
        p
        for p in available_pool
        if p.player_id != target.player_id
        and _roles_compatible(p, target, ruleset)
    ]

    bid_cap = max_affordable_bid(state, participant_id) if participant_id else None

    if not same_role:
        role_label = (
            "/".join(sorted(player_role_set(target, ruleset)))
            if ruleset == "MANTRA"
            else target.role
        )
        logger.info(
            "alternatives_none role=%s target=%s reason=role_exhausted_or_empty",
            role_label,
            target.player_id,
        )
        return AlternativeSuggestion(
            target_player_id=target.player_id,
            low_cost_alternative=None,
            closest_alternative=None,
            reason_if_none=f"reparto {role_label} esaurito o senza alternative",
            diversified_alternatives=(),
            max_affordable_bid=bid_cap,
            strategy_price_cap=None,
        )

    # Pre-calcola gli expected_price aggiornati (lazy, una sola volta).
    expected_prices: dict[str, float] = {
        p.player_id: project_price_for_player(state, p) for p in same_role
    }

    low_cost = _select_low_cost(
        target=target,
        candidates=same_role,
        expected_prices=expected_prices,
        config=config,
        valuation_mode=valuation_mode,
        apply_reliability_weight=apply_rw,
        risk_aversion=risk_av,
    )
    closest = _select_closest(
        target=target,
        candidates=same_role,
        expected_prices=expected_prices,
        valuation_mode=valuation_mode,
        apply_reliability_weight=apply_rw,
        risk_aversion=risk_av,
    )

    exclude = set()
    if low_cost is not None:
        exclude.add(low_cost.player_id)
    if closest is not None:
        exclude.add(closest.player_id)
    diversified: tuple = ()
    if diversify:
        diversified = tuple(
            pareto_diversify(
                same_role,
                expected_prices,
                valuation_mode,
                max_points=max_diversified,
                exclude_ids=exclude,
                apply_reliability_weight=apply_rw,
                risk_aversion=risk_av,
            )
        )

    target_price = project_price_for_player(state, target)
    strat_cap = strategy_price_cap(target, target_price, strategy_name)

    logger.info(
        "alternatives_suggested target=%s low_cost=%s closest=%s diversified=%d "
        "max_bid=%s strat_cap=%s",
        target.player_id,
        low_cost.player_id if low_cost else None,
        closest.player_id if closest else None,
        len(diversified),
        bid_cap,
        strat_cap,
    )

    return AlternativeSuggestion(
        target_player_id=target.player_id,
        low_cost_alternative=low_cost,
        closest_alternative=closest,
        reason_if_none=None,
        diversified_alternatives=diversified,
        max_affordable_bid=bid_cap,
        strategy_price_cap=strat_cap,
    )


# ---------------------------------------------------------------------------
# Internal selection rules
# ---------------------------------------------------------------------------


def _select_low_cost(
    target: Player,
    candidates: list[Player],
    expected_prices: dict[str, float],
    config: AlternativesConfig,
    valuation_mode: ValuationMode = ValuationMode.PER_MATCH_RATING,
    *,
    apply_reliability_weight: bool = True,
    risk_aversion: float = 0.0,
) -> Player | None:
    """Seleziona l'alternativa low-cost.

    Filtra i candidati il cui ``expected_price`` cade sotto la soglia
    ``low_cost_percentile`` del ruolo, e tra questi sceglie quello con
    miglior rapporto ``projected_score / expected_price``.

    Difensivamente, se nessun candidato cade sotto la soglia, restituisce
    ``None`` (meglio non suggerire un "low-cost" che in realtà non lo è).
    """
    if not candidates:
        return None

    # Soglia di expected_price calcolata come percentile del ruolo.
    sorted_prices = sorted(
        (expected_prices[p.player_id] for p in candidates),
        reverse=False,
    )
    n = len(sorted_prices)
    # Indice del quantile (inclusivo).  Es. low_cost_percentile=0.4, n=10
    # -> indice floor(0.4 * 9) = 3 -> sorted_prices[3] è il 40-esimo.
    quantile_idx = min(n - 1, int(config.low_cost_percentile * (n - 1)))
    price_threshold = sorted_prices[quantile_idx]

    eligible = [
        p
        for p in candidates
        if expected_prices[p.player_id] <= price_threshold + 1e-9
    ]
    if not eligible:
        return None

    def _score(p: Player) -> float:
        ep = expected_prices[p.player_id]
        if ep <= 0.0:
            return float("-inf")
        return _get_player_score(
            p, valuation_mode,
            apply_reliability_weight=apply_reliability_weight,
            risk_aversion=risk_aversion,
        ) / ep

    return max(eligible, key=_score)


def _select_closest(
    target: Player,
    candidates: list[Player],
    expected_prices: dict[str, float],
    valuation_mode: ValuationMode = ValuationMode.PER_MATCH_RATING,
    *,
    apply_reliability_weight: bool = True,
    risk_aversion: float = 0.0,
) -> Player | None:
    """Seleziona l'alternativa per affinità di rendimento.

    Restituisce il candidato con score più vicino al target (distanza
    assoluta minima); a parità di distanza, sceglie il costo atteso più
    basso come tie-break.
    """
    if not candidates:
        return None

    target_score = _get_player_score(
        target, valuation_mode,
        apply_reliability_weight=apply_reliability_weight,
        risk_aversion=risk_aversion,
    )

    def _sort_key(p: Player) -> tuple[float, float]:
        dist = abs(_get_player_score(
            p, valuation_mode,
            apply_reliability_weight=apply_reliability_weight,
            risk_aversion=risk_aversion,
        ) - target_score)
        ep = expected_prices[p.player_id]
        return (dist, ep)

    return min(candidates, key=_sort_key)
