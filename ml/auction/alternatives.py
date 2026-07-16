"""Pure function: alternative suggestions for a target player.

Given a target ``Player`` and the current ``AuctionState``, the function
returns two candidates restricted to the **same role**:

* ``low_cost_alternative`` - the available player with the best ratio
  ``projected_score / expected_price`` among those whose ``expected_price``
  falls under the configured percentile threshold of the role.
* ``closest_alternative`` - the available player with ``projected_score``
  closest (absolute distance) to the target; ties broken by lower
  ``expected_price``.

Both alternatives are validated to belong to the same role as the target.
If the role has no available candidates, the function returns explicit
``None`` results with a diagnostic ``reason_if_none`` (per spec §5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ml.auction.models import (
    AlternativeSuggestion,
    AlternativesConfig,
    AuctionState,
)
from ml.auction.price_drift import project_price_for_player
from ml.optimizer.models import Player

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

__all__ = ["suggest_alternatives"]


def suggest_alternatives(
    target: Player,
    available_pool: list[Player],
    state: AuctionState,
    config: AlternativesConfig,
) -> AlternativeSuggestion:
    """Suggerisce due alternative nello stesso ruolo del ``target``.

    Parameters
    ----------
    target:
        Giocatore bersaglio (può essere già stato assegnato — la firma
        ammette entrambi i casi).  Ne viene letto solo il ruolo.
    available_pool:
        Sottoinsieme del pool di giocatori *non ancora assegnati* e con
        ``role == target.role``.  L'orchestratore è tenuto a filtrare
        prima; in ogni caso la funzione rifiuta giocatori di ruoli
        diversi.
    state:
        Stato corrente dell'asta, usato per calcolare gli ``expected_price``
        aggiornati.
    config:
        Configurazione delle soglie (soprattutto ``low_cost_percentile``).

    Returns
    -------
    :class:`AlternativeSuggestion` con i due candidati.  Se il ruolo è
    esaurito, ``low_cost_alternative`` e ``closest_alternative`` sono
    entrambi ``None`` e ``reason_if_none`` spiega il motivo.
    """
    same_role = [
        p
        for p in available_pool
        if p.role == target.role and p.player_id != target.player_id
    ]

    if not same_role:
        logger.info(
            "alternatives_none role=%s target=%s reason=role_exhausted_or_empty",
            target.role,
            target.player_id,
        )
        return AlternativeSuggestion(
            target_player_id=target.player_id,
            low_cost_alternative=None,
            closest_alternative=None,
            reason_if_none=f"reparto {target.role} esaurito o senza alternative",
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
    )
    closest = _select_closest(
        target=target,
        candidates=same_role,
        expected_prices=expected_prices,
    )

    logger.info(
        "alternatives_suggested target=%s low_cost=%s closest=%s",
        target.player_id,
        low_cost.player_id if low_cost else None,
        closest.player_id if closest else None,
    )

    return AlternativeSuggestion(
        target_player_id=target.player_id,
        low_cost_alternative=low_cost,
        closest_alternative=closest,
        reason_if_none=None,
    )


# ---------------------------------------------------------------------------
# Internal selection rules
# ---------------------------------------------------------------------------


def _select_low_cost(
    target: Player,
    candidates: list[Player],
    expected_prices: dict[str, float],
    config: AlternativesConfig,
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

    # Miglior rapporto score/expected_price.  Difesa contro expected=0
    # (impossibile in pratica, ma controlliamo per evitare ZeroDivisionError).
    def _score(p: Player) -> float:
        ep = expected_prices[p.player_id]
        if ep <= 0.0:
            return float("-inf")
        return p.projected_score / ep

    return max(eligible, key=_score)


def _select_closest(
    target: Player,
    candidates: list[Player],
    expected_prices: dict[str, float],
) -> Player | None:
    """Seleziona l'alternativa per affinità di rendimento.

    Restituisce il candidato con ``projected_score`` più vicino al
    target (distanza assoluta minima); a parità di distanza, sceglie il
    costo atteso più basso come tie-break.
    """
    if not candidates:
        return None

    def _sort_key(p: Player) -> tuple[float, float]:
        dist = abs(p.projected_score - target.projected_score)
        ep = expected_prices[p.player_id]
        return (dist, ep)

    return min(candidates, key=_sort_key)
