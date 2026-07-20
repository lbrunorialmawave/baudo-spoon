"""Data models for the ILP-based squad optimizer.

All models are :func:`dataclasses.dataclass` ``frozen=True`` so the
optimizer pipeline is referentially transparent and trivially hashable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

from ml.mantra.roles import ALL_ROLES as _MANTRA_ALL_ROLES

__all__ = [
    "Role",
    "RulesetType",
    "StrategyName",
    "ROLE_QUOTAS",
    "MANTRA_DEFAULT_QUOTAS",
    "TOTAL_SQUAD_SIZE",
    "DEFAULT_BUDGET",
    "SOLVER_STATUS_OPTIMAL",
    "SOLVER_STATUS_INFEASIBLE",
    "SOLVER_STATUS_TIMEOUT",
    "SOLVER_STATUS_UNBOUNDED",
    "SOLVER_STATUS_ERROR",
    "Player",
    "Formation",
    "InflationConfig",
    "StrategyProfile",
    "OptimizationConfig",
    "OptimizationResult",
    "MultiStrategyResult",
]

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

Role = Literal["P", "D", "C", "A"]
"""Fantacalcio role codes: P=portiere, D=difensore, C=centrocampista, A=attaccante."""

RulesetType = Literal["CLASSIC", "MANTRA"]
"""Ruleset: CLASSIC (4 roles, single assignment) or MANTRA (12 roles, multi-slot)."""

StrategyName = Literal[
    "BALANCED", "SUPER_DEFENSIVE", "SUPER_OFFENSIVE", "MIXED"
]

# Quotas in rosa (Fantacalcio classico).
ROLE_QUOTAS: Final[dict[Role, int]] = {
    "P": 3,
    "D": 8,
    "C": 8,
    "A": 6,
}
"""Quota fissa di giocatori per ruolo in una rosa da 25."""

# Default quotas for MANTRA (12 roles, total 25).
# MANTRA leagues vary in quota definitions; callers can override via
# OptimizationConfig.mantra_role_quotas.
# Por=3, defenders(Dc+B+Dd+Ds)=8, midfield(E+M+C)=8, attack(T+W+A+Pc)=6.
MANTRA_DEFAULT_QUOTAS: Final[dict[str, int]] = {
    "Por": 3,
    "Dc": 3, "B": 2, "Dd": 2, "Ds": 1,
    "E": 1, "M": 2, "C": 5,
    "T": 1, "W": 1, "A": 2, "Pc": 2,
}

TOTAL_SQUAD_SIZE: Final[int] = 25
"""Numero totale di giocatori in rosa."""

DEFAULT_BUDGET: Final[int] = 500
"""Budget di default in crediti (Fantacalcio classico)."""

SOLVER_STATUS_OPTIMAL: Final[str] = "OPTIMAL"
SOLVER_STATUS_INFEASIBLE: Final[str] = "INFEASIBLE"
SOLVER_STATUS_TIMEOUT: Final[str] = "TIMEOUT"
SOLVER_STATUS_UNBOUNDED: Final[str] = "UNBOUNDED"
SOLVER_STATUS_ERROR: Final[str] = "ERROR"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Player:
    """Singolo giocatore nel pool.

    ``player_id`` è l'identificativo univoco (es. id FotMob), **mai** il
    ``name``, perché il dominio gestisce omonimie.
    """

    player_id: str
    name: str
    role: Role
    real_team: str
    cost: int
    projected_score: float
    reliability_weight: float | None = None
    # MANTRA only: set of Mantra role codes this player can fill (e.g. {"Dd", "E"}).
    # Must be non-empty when ruleset == "MANTRA". Ignored for CLASSIC.
    eligible_roles: frozenset[str] = field(default_factory=frozenset)
    # Standard deviation of the ensemble prediction. When present, used by the
    # risk-adjusted objective: score -= risk_aversion * prediction_std.
    prediction_std: float | None = None
    # Picco / listino_cost from pilastro4 historical auction data. When present,
    # blended with role_percentile in estimate_effective_cost as a player-specific prior.
    historical_overpay_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.player_id:
            raise ValueError("Player.player_id must be non-empty")
        if not self.name:
            raise ValueError("Player.name must be non-empty")
        if self.role not in ROLE_QUOTAS:
            raise ValueError(
                f"Player.role must be one of {tuple(ROLE_QUOTAS)}, got {self.role!r}"
            )
        if not self.real_team:
            raise ValueError("Player.real_team must be non-empty")
        if self.cost < 0:
            raise ValueError(f"Player.cost must be >= 0, got {self.cost}")
        if self.projected_score < 0:
            raise ValueError(
                f"Player.projected_score must be >= 0, got {self.projected_score}"
            )
        if self.reliability_weight is not None and self.reliability_weight < 0:
            raise ValueError(
                "Player.reliability_weight must be >= 0 if provided, "
                f"got {self.reliability_weight}"
            )
        if self.prediction_std is not None and self.prediction_std < 0:
            raise ValueError(
                "Player.prediction_std must be >= 0 if provided, "
                f"got {self.prediction_std}"
            )
        if self.historical_overpay_ratio is not None and self.historical_overpay_ratio < 0:
            raise ValueError(
                "Player.historical_overpay_ratio must be >= 0 if provided, "
                f"got {self.historical_overpay_ratio}"
            )


@dataclass(frozen=True)
class Formation:
    """Modulo tattico: numero di difensori, centrocampisti, attaccanti titolari.

    Il portiere titolare è sempre 1; la rosa contiene 3 P in totale.
    """

    label: str
    defenders: int
    midfielders: int
    forwards: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Formation.label must be non-empty")
        for field_name in ("defenders", "midfielders", "forwards"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(
                    f"Formation.{field_name} must be >= 0, got {value}"
                )


@dataclass(frozen=True)
class InflationConfig:
    """Configurazione della funzione di inflazione del costo d'asta.

    Tutti i parametri sono configurabili; nessun valore hardcoded nel solver.
    """

    inflation_percentile_threshold: float = 0.7
    """Soglia di percentile (in [0,1]) sotto la quale l'inflazione è nulla."""

    max_inflation_multiplier: float = 1.6
    """Cap massimo al moltiplicatore (costo_effettivo / costo_listino)."""

    base_inflation_rate: float = 0.05
    """Tasso base di inflazione per partecipante oltre la baseline (top tier)."""

    baseline_participants: int = 8
    """Numero di partecipanti di riferimento senza inflazione."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.inflation_percentile_threshold <= 1.0:
            raise ValueError(
                "inflation_percentile_threshold must be in [0,1], got "
                f"{self.inflation_percentile_threshold}"
            )
        if self.max_inflation_multiplier < 1.0:
            raise ValueError(
                "max_inflation_multiplier must be >= 1.0, got "
                f"{self.max_inflation_multiplier}"
            )
        if self.base_inflation_rate < 0:
            raise ValueError(
                "base_inflation_rate must be >= 0, got "
                f"{self.base_inflation_rate}"
            )
        if self.baseline_participants < 1:
            raise ValueError(
                "baseline_participants must be >= 1, got "
                f"{self.baseline_participants}"
            )


@dataclass(frozen=True)
class StrategyProfile:
    """Profilo di strategia: pesi di ruolo e vincoli soft."""

    name: StrategyName
    role_weight: dict[str, float]
    min_budget_share_by_roles: tuple[frozenset[str], float] | None = None
    max_top_tier_players: int | None = None
    top_tier_cost_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StrategyProfile.name must be non-empty")
        missing = set(ROLE_QUOTAS) - set(self.role_weight)
        if missing:
            raise ValueError(
                f"StrategyProfile.role_weight missing roles: {sorted(missing)}"
            )
        if any(w < 0 for w in self.role_weight.values()):
            raise ValueError("StrategyProfile.role_weight values must be >= 0")
        if self.min_budget_share_by_roles is not None:
            roles, share = self.min_budget_share_by_roles
            if not roles:
                raise ValueError(
                    "min_budget_share_by_roles roles set must be non-empty"
                )
            if not 0.0 <= share <= 1.0:
                raise ValueError(
                    "min_budget_share_by_roles share must be in [0,1], "
                    f"got {share}"
                )
            for r in roles:
                if r not in ROLE_QUOTAS:
                    raise ValueError(
                        f"min_budget_share_by_roles has invalid role {r!r}"
                    )
        if self.max_top_tier_players is not None and self.max_top_tier_players < 0:
            raise ValueError(
                "max_top_tier_players must be >= 0, got "
                f"{self.max_top_tier_players}"
            )
        if (
            self.max_top_tier_players is not None
            and self.top_tier_cost_threshold is None
        ):
            raise ValueError(
                "top_tier_cost_threshold is required when max_top_tier_players "
                "is set"
            )
        if self.top_tier_cost_threshold is not None and self.top_tier_cost_threshold < 0:
            raise ValueError(
                "top_tier_cost_threshold must be >= 0, got "
                f"{self.top_tier_cost_threshold}"
            )


@dataclass(frozen=True)
class OptimizationConfig:
    """Configurazione completa della singola run di ottimizzazione."""

    budget: int
    formations: list[Formation]
    num_participants: int
    max_players_per_team: int = 4
    big_teams: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"Inter", "Milan", "Juventus", "Napoli"}
        )
    )
    big_teams_cap: int = 10
    min_distinct_teams: int = 12
    inflation_config: InflationConfig = field(default_factory=InflationConfig)
    strategies: tuple[StrategyProfile, ...] = field(
        default_factory=tuple  # overridden in __post_init__
    )
    solver_timeout_seconds: int = 30
    # Maximum share of total budget that can be spent on a single player
    # (effective cost). Prevents the solver from concentrating budget on one
    # player who might be sniped in auction before your turn.
    max_single_player_budget_share: float = 0.30
    # Player IDs that must / must not appear in the squad.
    must_include: frozenset[str] = field(default_factory=frozenset)
    exclude: frozenset[str] = field(default_factory=frozenset)
    # Ruleset: CLASSIC (4 roles) or MANTRA (12 roles with multi-slot eligibility).
    # Default CLASSIC preserves existing behaviour for all current callers.
    ruleset: RulesetType = "CLASSIC"
    # MANTRA only: role quotas for the 12 Mantra roles. Defaults to
    # MANTRA_DEFAULT_QUOTAS when None and ruleset == "MANTRA".
    mantra_role_quotas: dict[str, int] | None = None
    # When set, the solver enforces that the squad can field this specific module.
    # If None, no formation constraint is applied — formation_feasibility in the
    # result is purely informational (post-hoc check, not a solver guarantee).
    preferred_formation: Formation | None = None
    # Risk-aversion coefficient for the objective: effective_score = projected_score
    # - risk_aversion * prediction_std. Default 0.0 = risk-neutral (backward compat).
    # Meaningful only when Player.prediction_std is populated by the ensemble.
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"OptimizationConfig.budget must be > 0, got {self.budget}")
        if not self.formations:
            raise ValueError("OptimizationConfig.formations must be non-empty")
        if self.num_participants < 1:
            raise ValueError(
                f"OptimizationConfig.num_participants must be >= 1, got "
                f"{self.num_participants}"
            )
        if self.max_players_per_team < 1:
            raise ValueError(
                f"max_players_per_team must be >= 1, got {self.max_players_per_team}"
            )
        if self.big_teams_cap < 0:
            raise ValueError(
                f"big_teams_cap must be >= 0, got {self.big_teams_cap}"
            )
        if self.min_distinct_teams < 1:
            raise ValueError(
                f"min_distinct_teams must be >= 1, got {self.min_distinct_teams}"
            )
        if self.solver_timeout_seconds <= 0:
            raise ValueError(
                f"solver_timeout_seconds must be > 0, got {self.solver_timeout_seconds}"
            )
        if not 0.0 < self.max_single_player_budget_share <= 1.0:
            raise ValueError(
                "max_single_player_budget_share must be in (0, 1], got "
                f"{self.max_single_player_budget_share}"
            )
        conflict = self.must_include & self.exclude
        if conflict:
            raise ValueError(
                f"player_ids appear in both must_include and exclude: {sorted(conflict)}"
            )
        if self.ruleset == "MANTRA" and self.mantra_role_quotas is not None:
            if sum(self.mantra_role_quotas.values()) != TOTAL_SQUAD_SIZE:
                raise ValueError(
                    f"mantra_role_quotas must sum to {TOTAL_SQUAD_SIZE}, got "
                    f"{sum(self.mantra_role_quotas.values())}"
                )
        if self.risk_aversion < 0:
            raise ValueError(
                f"risk_aversion must be >= 0, got {self.risk_aversion}"
            )
        # Default strategies are injected lazily to avoid circular import
        # between strategies.py and this module.
        if not self.strategies:
            object.__setattr__(self, "strategies", _default_strategies())


def _default_strategies() -> tuple[StrategyProfile, ...]:
    """Lazy import for the default 4 strategies."""
    # Imported here to avoid module-level circular dependency.
    from ml.optimizer.strategies import default_strategies as _ds

    return _ds()


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """Risultato di una singola strategia di ottimizzazione."""

    strategy_name: str
    status: Literal["OPTIMAL", "INFEASIBLE", "TIMEOUT", "UNBOUNDED", "ERROR"]
    squad: list[Player]
    total_nominal_cost: int
    total_effective_cost: float
    total_projected_score: float
    budget_residual: float
    role_breakdown: dict[str, int]
    team_breakdown: dict[str, int]
    distinct_teams_count: int
    big_teams_players_count: int
    formation_feasibility: dict[str, bool]
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiStrategyResult:
    """Risultato aggregato delle 4 strategie."""

    results: dict[str, OptimizationResult]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("MultiStrategyResult.results must be non-empty")
