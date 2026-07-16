"""Pure functions for dynamic price drift (EWMA + tier classification).

This module is intentionally free of side effects on external state: it
operates on plain Python values and returns new values where possible.
The mutable exception is :func:`update_price_index`, which updates the
``price_index`` mapping in place (to avoid copying a nested dict at every
assignment) but returns a deep snapshot of the pre-update state for
deterministic undo support.

The EWMA update is the core of the live price drift model: after each
recorded assignment, the (role, tier) index is updated as an exponential
moving average of the ratio between the actual price paid and the expected
price prior to the update.  A small, multiplicative spillover is then
propagated to adjacent tiers of the same role, and an optional
(disabled-by-default) cross-role spillover is also available as a hook.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from ml.auction.models import (
    ADJACENT_TIERS,
    ALL_TIERS,
    AuctionConfig,
    AuctionState,
    MarketDriftConfig,
    Role,
    Tier,
)
from ml.optimizer.inflation import estimate_effective_cost
from ml.optimizer.models import Player

logger = logging.getLogger(__name__)

__all__ = [
    "classify_tier",
    "build_initial_price_index",
    "compute_baseline_cost",
    "compute_expected_price",
    "update_price_index",
    "apply_cross_role_spillover",
    "clamp_index",
    "get_current_projection",
    "project_price_for_player",
]


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def classify_tier(percentile: float, config: MarketDriftConfig) -> Tier:
    """Classifica un giocatore nel tier corrispondente al suo percentile.

    ``percentile`` deve essere in ``[0, 1]``; valori fuori range vengono
    clamppati difensivamente per evitare sorprese in runtime.
    """
    pct = min(1.0, max(0.0, float(percentile)))
    low_thr, top_thr = config.tier_thresholds
    if pct >= top_thr:
        return "TOP"
    if pct >= low_thr:
        return "MID"
    return "LOW"


# ---------------------------------------------------------------------------
# Price index lifecycle
# ---------------------------------------------------------------------------


def build_initial_price_index(
    config: AuctionConfig,
) -> dict[Role, dict[Tier, float]]:
    """Inizializza ``price_index`` a 1.0 per ogni combinazione (ruolo, tier)."""
    out: dict[Role, dict[Tier, float]] = {}
    for role in config.role_quotas:
        out[role] = {tier: 1.0 for tier in ALL_TIERS}
    return out


def clamp_index(value: float, config: MarketDriftConfig) -> float:
    """Clamp difensivo di un singolo valore entro ``[min_index, max_index]``."""
    return min(config.max_index, max(config.min_index, value))


# ---------------------------------------------------------------------------
# Baseline / expected price
# ---------------------------------------------------------------------------


def compute_baseline_cost(
    player: Player,
    role_percentile: float,
    config: AuctionConfig,
) -> float:
    """Restituisce il ``baseline_cost`` di un giocatore per il price drift.

    Il listino ``player.cost`` è storicamente tarato su un fantacalcio a
    ``config.reference_budget`` crediti/squadra.  Se l'asta corrente usa un
    ``config.budget_initial`` diverso, il costo base viene riproporzionato
    per il fattore ``budget_initial / reference_budget`` in modo che il
    ``price_index`` EWMA parta da un'aspettativa coerente con il potere
    d'acquisto reale della lega.  Se ``AuctionConfig.use_inflation_baseline``
    è ``True``, l'inflazione statica dell'ottimizzatore rosa
    (:func:`estimate_effective_cost`) viene applicata sul listino già
    riproporzionato.
    """
    scale = config.budget_initial / config.reference_budget
    scaled_cost = float(player.cost) * scale
    if not config.use_inflation_baseline:
        return scaled_cost

    inflation_cfg = config.inflation_config
    if inflation_cfg is None:  # pragma: no cover - difesa in profondità
        raise ValueError(
            "AuctionConfig.inflation_config must be set when "
            "use_inflation_baseline=True"
        )
    # L'oggetto è validato a runtime dall'orchestratore; il cast è sicuro.
    scaled_player = replace(player, cost=int(round(scaled_cost)))
    return estimate_effective_cost(
        player=scaled_player,
        role_percentile=role_percentile,
        num_participants=config.num_participants,
        config=inflation_cfg,  # type: ignore[arg-type]
    )


def compute_expected_price(
    player: Player,
    role_percentile: float,
    role: Role,
    tier: Tier,
    price_index: dict[Role, dict[Tier, float]],
    config: AuctionConfig,
) -> float:
    """Prezzo atteso corrente = ``baseline_cost * price_index[role][tier]``."""
    baseline = compute_baseline_cost(player, role_percentile, config)
    idx = price_index[role][tier]
    return baseline * idx


# ---------------------------------------------------------------------------
# EWMA update
# ---------------------------------------------------------------------------


def update_price_index(
    role: Role,
    tier: Tier,
    actual_price: float,
    expected_price: float,
    price_index: dict[Role, dict[Tier, float]],
    config: MarketDriftConfig,
) -> tuple[float, float, dict[Role, dict[Tier, float]]]:
    """Aggiorna ``price_index`` in-place e ritorna ``(before, after, snapshot)``.

    Parameters
    ----------
    role, tier:
        Coordinate del tier aggiornato dall'assegnazione registrata.
    actual_price:
        Prezzo reale pagato all'asta (inserito dall'operatore).
    expected_price:
        Prezzo atteso *prima* dell'aggiornamento
        (``baseline_cost * price_index[role][tier]`` al momento del record).
    price_index:
        Mappa mutabile ``{role: {tier: index}}`` aggiornata in place.
    config:
        Coefficienti del modello EWMA.

    Returns
    -------
    ``(index_before, index_after, snapshot_before)``:

    * ``index_before`` / ``index_after`` sono i valori di
      ``price_index[role][tier]`` rispettivamente prima e dopo l'aggiornamento.
    * ``snapshot_before`` è una copia profonda di *tutto* ``price_index``
      prima dell'aggiornamento, utile per implementare l'undo deterministico.

    Note
    ----
    * L'aggiornamento è un EWMA standard:
      ``new = (1 - alpha) * old + alpha * (actual / expected)``.
    * Lo spillover verso i tier adiacenti è attenuato:
      ``adj_new = adj_old * (1 + spillover * (ratio - 1))``.
    * Tutti i valori sono clamppati in ``[min_index, max_index]`` ad ogni
      step per garantire che derive prolungate non esplodano.
    * Lo spillover cross-ruolo è applicato come hook opzionale quando
      ``spillover_cross_role > 0``.
    """
    if expected_price <= 0.0:
        raise ValueError(
            f"expected_price must be > 0 for EWMA, got {expected_price}"
        )
    if actual_price < 0.0:
        raise ValueError(f"actual_price must be >= 0, got {actual_price}")
    if role not in price_index:
        raise ValueError(f"role {role!r} not present in price_index")
    if tier not in price_index[role]:
        raise ValueError(f"tier {tier!r} not present in price_index[{role!r}]")

    # Snapshot dell'intero price_index per consentire l'undo deterministico.
    snapshot_before: dict[Role, dict[Tier, float]] = {
        r: dict(tiers) for r, tiers in price_index.items()
    }

    index_before = price_index[role][tier]
    ratio = actual_price / expected_price

    # 1) Aggiornamento EWMA del tier principale.
    new_main = (1.0 - config.alpha) * index_before + config.alpha * ratio
    price_index[role][tier] = clamp_index(new_main, config)
    index_after = price_index[role][tier]

    # 2) Spillover attenuato ai tier adiacenti dello stesso ruolo.
    if config.spillover_adjacent_tier > 0.0:
        for adj_tier in ADJACENT_TIERS[tier]:
            adj_old = price_index[role][adj_tier]
            adj_new = adj_old * (1.0 + config.spillover_adjacent_tier * (ratio - 1.0))
            price_index[role][adj_tier] = clamp_index(adj_new, config)

    # 3) Hook di spillover cross-ruolo (disattivato di default).
    if config.spillover_cross_role > 0.0:
        apply_cross_role_spillover(
            source_role=role,
            source_tier=tier,
            ratio=ratio,
            price_index=price_index,
            config=config,
        )

    logger.info(
        "price_index_updated role=%s tier=%s actual=%.2f expected=%.2f "
        "index_before=%.4f index_after=%.4f",
        role,
        tier,
        actual_price,
        expected_price,
        index_before,
        index_after,
    )

    return index_before, index_after, snapshot_before


def apply_cross_role_spillover(
    source_role: Role,
    source_tier: Tier,
    ratio: float,
    price_index: dict[Role, dict[Tier, float]],
    config: MarketDriftConfig,
) -> None:
    """Applica spillover allo stesso tier degli *altri* ruoli.

    Implementato come hook disattivato di default.  Quando attivo
    (``spillover_cross_role > 0``), propaga lo stesso shock
    ``(ratio - 1)`` allo stesso ``tier`` di tutti gli altri ruoli, con
    intensità attenuata dal coefficiente.  È una scelta ragionevole per
    modellare l'effetto "i top di un ruolo costano di più, quindi anche i
    top di altri ruoli subiscono una pressione indiretta"; l'orchestratore
    può usare un coefficiente molto piccolo (es. ``0.05``) per un effetto
    appena percettibile.

    La funzione muta ``price_index`` in place; ogni valore viene clampato.
    """
    for other_role, tiers in price_index.items():
        if other_role == source_role:
            continue
        if source_tier not in tiers:
            continue
        old = tiers[source_tier]
        new_value = old * (1.0 + config.spillover_cross_role * (ratio - 1.0))
        tiers[source_tier] = clamp_index(new_value, config)


# ---------------------------------------------------------------------------
# Live projection
# ---------------------------------------------------------------------------


def project_price_for_player(
    state: AuctionState,
    player: Player,
) -> float:
    """Prezzo atteso aggiornato per un giocatore non ancora assegnato.

    Funzione di supporto richiamata da :func:`get_current_projection` ed
    esposta anche come utility di test.
    """
    role = player.role
    percentile = state.role_percentile_map.get(player.player_id, 0.0)
    tier = classify_tier(percentile, state.config.market_drift_config)
    return compute_expected_price(
        player=player,
        role_percentile=percentile,
        role=role,
        tier=tier,
        price_index=state.price_index,
        config=state.config,
    )


def get_current_projection(
    state: AuctionState,
    player_id: str,
    pool: list[Player],
) -> float:
    """Prezzo atteso aggiornato per ``player_id`` interrogabile in ogni momento.

    Parameters
    ----------
    state:
        Stato corrente dell'asta.
    player_id:
        Identificativo del giocatore (deve essere ancora nel pool).
    pool:
        Pool completo da cui cercare il giocatore (l'operatore deve passare
        il pool originale, non ``state.available_pool``, perché il giocatore
        è tipicamente non ancora assegnato e quindi ancora disponibile).
    """
    player = _find_player(pool, player_id)
    if player is None:
        raise ValueError(f"player_id {player_id!r} not found in pool")
    return project_price_for_player(state, player)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_player(pool: list[Player], player_id: str) -> Player | None:
    for p in pool:
        if p.player_id == player_id:
            return p
    return None
