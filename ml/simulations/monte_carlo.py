"""Monte Carlo season simulator.

Sampling method: bootstrap residuals from historical prediction errors.
  - For each player, draw N_SIMULATIONS residuals from the empirical distribution
    of (actual_fantavoto - predicted_fantavoto) for that role.
  - Simulated season score = predicted_score + resampled_residual
  - If per-player residuals < MIN_RESIDUALS, fall back to role-level residuals.
  - If role-level residuals also absent, use N(0, DEFAULT_STD) parametric fallback
    (documented explicitly as uncalibrated).

N_SIMULATIONS default: 1000 (configurable per call).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

log = logging.getLogger(__name__)

N_SIMULATIONS: int = 1000
MIN_RESIDUALS: int = 10
DEFAULT_STD: float = 0.8  # parametric fallback std (uncalibrated)


@dataclass(frozen=True)
class SimulationResult:
    """Per-player Monte Carlo simulation output."""

    player_id: str
    n_simulations: int
    mean_score: float
    std_score: float
    p10_score: float          # 10th percentile (downside)
    p25_score: float
    p50_score: float          # median
    p75_score: float
    p90_score: float          # upside
    upside_potential: float   # p90 - p50
    downside_risk: float      # p50 - p10
    simulated_scores: np.ndarray  # shape (n_simulations,)
    sampling_method: str      # "bootstrap_player", "bootstrap_role", "parametric"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimulationResult):
            return NotImplemented
        return self.player_id == other.player_id and self.n_simulations == other.n_simulations

    def __hash__(self) -> int:
        return hash((self.player_id, self.n_simulations))


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for fantasy football season scores.

    Usage::

        sim = MonteCarloSimulator(random_seed=42)
        sim.fit(residuals)  # residuals: list of {player_id, role, residual}
        results = sim.simulate_many(players, n_simulations=1000)

    Args:
        random_seed: RNG seed for reproducibility.
        clip_low: Minimum simulated fantavoto score (Fantacalcio minimum is ~1.0).
        clip_high: Maximum simulated fantavoto score (practical cap ~10.0).
    """

    def __init__(
        self,
        random_seed: int = 42,
        clip_low: float = 1.0,
        clip_high: float = 10.0,
    ) -> None:
        self.random_seed = random_seed
        self.clip_low = clip_low
        self.clip_high = clip_high
        self._player_residuals: dict[str, np.ndarray] = {}
        self._role_residuals: dict[str, np.ndarray] = {}
        self._is_fitted = False

    def fit(self, residuals: list[dict]) -> "MonteCarloSimulator":
        """Build residual distributions from historical prediction errors.

        Args:
            residuals: List of dicts with keys player_id (str), role (str),
                residual (float = actual - predicted).

        Returns:
            self, for chaining.
        """
        player_res: dict[str, list[float]] = defaultdict(list)
        role_res: dict[str, list[float]] = defaultdict(list)

        for r in residuals:
            pid = str(r["player_id"])
            role = str(r["role"])
            val = float(r["residual"])
            player_res[pid].append(val)
            role_res[role].append(val)

        self._player_residuals = {k: np.array(v) for k, v in player_res.items()}
        self._role_residuals = {k: np.array(v) for k, v in role_res.items()}
        self._is_fitted = True

        log.info(
            "MonteCarloSimulator fitted: %d players, %d roles.",
            len(self._player_residuals),
            len(self._role_residuals),
        )
        return self

    def simulate(
        self,
        player_id: str,
        predicted_score: float,
        role: str,
        n_simulations: int = N_SIMULATIONS,
    ) -> SimulationResult:
        """Simulate one player's season scores.

        Args:
            player_id: Unique player identifier.
            predicted_score: Point prediction from the ensemble.
            role: Fantacalcio role code (P/D/C/A).
            n_simulations: Number of synthetic seasons.

        Returns:
            SimulationResult with percentile distribution.
        """
        rng = np.random.default_rng(self.random_seed + abs(hash(player_id)) % (2**31))

        player_res = self._player_residuals.get(player_id)
        role_res = self._role_residuals.get(role)

        if player_res is not None and len(player_res) >= MIN_RESIDUALS:
            residuals_pool = player_res
            method = "bootstrap_player"
        elif role_res is not None and len(role_res) >= MIN_RESIDUALS:
            residuals_pool = role_res
            method = "bootstrap_role"
            log.debug(
                "Player '%s': insufficient residuals (%d < %d); using role '%s' pool.",
                player_id,
                len(player_res) if player_res is not None else 0,
                MIN_RESIDUALS,
                role,
            )
        else:
            # Parametric fallback — explicitly uncalibrated
            residuals_pool = rng.normal(0.0, DEFAULT_STD, size=n_simulations * 2)
            method = "parametric"
            log.warning(
                "Player '%s' role '%s': no residual data; using parametric fallback "
                "(uncalibrated, std=%.2f). Fit MonteCarloSimulator on real residuals "
                "before trusting these intervals.",
                player_id,
                role,
                DEFAULT_STD,
            )

        sampled = rng.choice(residuals_pool, size=n_simulations, replace=True)
        scores = np.clip(predicted_score + sampled, self.clip_low, self.clip_high)

        return SimulationResult(
            player_id=player_id,
            n_simulations=n_simulations,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            p10_score=float(np.percentile(scores, 10)),
            p25_score=float(np.percentile(scores, 25)),
            p50_score=float(np.percentile(scores, 50)),
            p75_score=float(np.percentile(scores, 75)),
            p90_score=float(np.percentile(scores, 90)),
            upside_potential=float(np.percentile(scores, 90) - np.percentile(scores, 50)),
            downside_risk=float(np.percentile(scores, 50) - np.percentile(scores, 10)),
            simulated_scores=scores,
            sampling_method=method,
        )

    def simulate_many(
        self,
        players: list[dict],
        n_simulations: int = N_SIMULATIONS,
    ) -> list[SimulationResult]:
        """Simulate all players. Each player gets its own RNG seed derived from player_id.

        Args:
            players: List of dicts with player_id, predicted_score, role.
            n_simulations: Number of synthetic seasons per player.

        Returns:
            List of SimulationResult, one per player.
        """
        return [
            self.simulate(
                str(p["player_id"]),
                float(p["predicted_score"]),
                str(p["role"]),
                n_simulations,
            )
            for p in players
        ]
