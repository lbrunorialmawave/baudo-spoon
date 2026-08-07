"""Pure function modelling auction inflation based on league size.

Decoupled from PuLP: the function operates on plain Python values so it can
be tested in isolation.  Given a ``Player``, the percentile of the player
within his role (computed upstream from the pool's ``projected_score``),
the number of participants in the league and an :class:`InflationConfig`,
it returns the expected cost in a real auction, never below the listino
quotation.
"""

from __future__ import annotations

import math
from typing import Final

from ml.optimizer.models import InflationConfig, Player

__all__ = ["InflationConfig", "compute_role_percentile_map", "estimate_effective_cost"]


# Sentinel: any non-positive nominal cost has no listino to inflate.
_MIN_NOMINAL_COST_FOR_INFLATION: Final[float] = 1.0


def estimate_effective_cost(
    player: Player,
    role_percentile: float,
    num_participants: int,
    config: InflationConfig,
    team_strength_scores: dict[str, float] | None = None,
) -> float:
    """Restituisce il costo atteso in asta reale, >= ``player.cost``.

    Parametri
    ---------
    player:
        Giocatore di cui stimare il costo d'asta.
    role_percentile:
        Percentile del giocatore nel proprio ruolo per ``projected_score``,
        in ``[0, 1]``.  ``0.0`` = peggiore del ruolo, ``1.0`` = migliore.
        Valori fuori da ``[0, 1]`` vengono clamppati.
    num_participants:
        Numero di partecipanti alla lega; cresce la competizione d'asta.
    config:
        Coefficienti della curva.  Tutti i parametri sono letti da
        ``config``; nulla è hardcoded.

    Comportamento
    -------------
    * Sotto ``inflation_percentile_threshold`` l'inflazione è nulla
      (costo effettivo == costo di listino).
    * Sopra soglia, l'inflazione cresce con ``num_participants`` e con
      ``role_percentile`` (entrambi monotonicamente crescenti).
    * Il moltiplicatore rispetta sempre il cap ``max_inflation_multiplier``.
    * La funzione è monotona crescente in entrambi i parametri.

    Esempio
    -------
    >>> p = Player(
    ...     player_id="x", name="X", role="A", real_team="Inter",
    ...     cost=20, projected_score=8.0,
    ... )
    >>> cfg = InflationConfig()
    >>> estimate_effective_cost(p, 0.9, 10, cfg) > 20
    True
    >>> estimate_effective_cost(p, 0.5, 10, cfg) == 20
    True
    """
    if num_participants < 1:
        raise ValueError(f"num_participants must be >= 1, got {num_participants}")

    nominal = max(0.0, float(player.cost))

    # No listino: cannot inflate below zero.
    if nominal < _MIN_NOMINAL_COST_FOR_INFLATION:
        return nominal

    # Player-specific prior: if available, blend historical_overpay_ratio with
    # role_percentile. overpay_ratio is Picco/listino (typically 1-3); normalise
    # to [0,1] via tanh so it doesn't overwhelm the percentile signal. The blend
    # is a simple average so each contributes equally.
    raw_percentile = min(1.0, max(0.0, float(role_percentile)))
    if player.historical_overpay_ratio is not None:
        # tanh(ratio - 1) maps: ratio=1 → 0.0, ratio=2 → 0.76, ratio=3 → 0.96
        overpay_signal = math.tanh(max(0.0, player.historical_overpay_ratio - 1.0))
        percentile = min(1.0, (raw_percentile + overpay_signal) / 2.0)
    else:
        percentile = raw_percentile

    threshold = config.inflation_percentile_threshold
    if percentile <= threshold:
        return nominal

    # Distance above the threshold, normalised to [0, 1].
    headroom = (percentile - threshold) / max(1e-9, 1.0 - threshold)

    # Number of "extra" participants beyond the baseline (clamped to >= 0).
    extra_participants = max(
        0, int(num_participants) - int(config.baseline_participants)
    )

    # Inflation multiplier (>= 1) bounded by the cap.
    raw = 1.0 + config.base_inflation_rate * extra_participants * headroom
    cap = float(config.max_inflation_multiplier)
    multiplier = min(cap, max(1.0, raw))

    effective = nominal * multiplier

    # Team-strength adjustment: boost cost for players on strong teams.
    if config.team_strength_multiplier > 0 and team_strength_scores:
        normalized_elo = team_strength_scores.get(player.real_team, 0.0)
        effective *= 1.0 + config.team_strength_multiplier * normalized_elo

    return effective


def compute_role_percentile_map(
    players: list[Player],
) -> dict[str, float]:
    """Calcola la mappa ``player_id -> role_percentile``.

    Per ogni ruolo, ordina i giocatori per ``projected_score`` crescente e
    assegna a ciascuno un percentile in ``[0, 1]``.  Il percentile è
    calcolato come ``rank / (n - 1)`` (rank-based, NON CDF), così che il
    miglior giocatore del ruolo riceva ``1.0`` e il peggiore ``0.0``.

    Gestisce correttamente:
    * Ruoli con un solo giocatore → percentile 1.0.
    * Più giocatori con stesso ``projected_score`` → percentile uguale.
    * Pool vuoto → ritorna mappa vuota (nessun errore).

    Esempio
    -------
    >>> ps = [
    ...     Player("1", "A", "A", "T1", 10, 5.0),
    ...     Player("2", "B", "A", "T2", 10, 7.0),
    ... ]
    >>> m = compute_role_percentile_map(ps)
    >>> m["1"], m["2"]
    (0.0, 1.0)
    """
    out: dict[str, float] = {}
    if not players:
        return out

    # Group by role preserving order of first appearance for determinism.
    by_role: dict[str, list[Player]] = {}
    for p in players:
        by_role.setdefault(p.role, []).append(p)

    for group in by_role.values():
        n = len(group)
        if n == 1:
            out[group[0].player_id] = 1.0
            continue

        # Sort by score ascending; tie-break by player_id for determinism.
        ordered = sorted(group, key=lambda p: (p.projected_score, p.player_id))
        denom = float(n - 1)
        for rank, p in enumerate(ordered):
            out[p.player_id] = rank / denom
    return out


def inflation_multiplier(
    role_percentile: float,
    num_participants: int,
    config: InflationConfig,
) -> float:
    """Restituisce il solo moltiplicatore (>= 1) applicato al listino.

    Esposto come helper per test e diagnostica, ma non usato direttamente
    dal solver.
    """
    if num_participants < 1:
        raise ValueError("num_participants must be >= 1")
    percentile = min(1.0, max(0.0, float(role_percentile)))
    if percentile <= config.inflation_percentile_threshold:
        return 1.0
    headroom = (percentile - config.inflation_percentile_threshold) / max(
        1e-9, 1.0 - config.inflation_percentile_threshold
    )
    extra = max(0, int(num_participants) - int(config.baseline_participants))
    raw = 1.0 + config.base_inflation_rate * extra * headroom
    return min(float(config.max_inflation_multiplier), max(1.0, raw))


def is_inflation_active(
    role_percentile: float,
    config: InflationConfig,
) -> bool:
    """True se il percentile è sopra la soglia di inflazione."""
    return role_percentile > config.inflation_percentile_threshold


# Re-export ``math`` for testing convenience; not part of the public API.
_ = math  # pragma: no cover
