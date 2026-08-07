"""Data models for the live auction tracker.

All models are :func:`dataclasses.dataclass` instances.  The schemas follow
the specification in the live auction tracker spec:

* Configuration objects (``MarketDriftConfig``, ``AlternativesConfig``,
  ``ParticipantSetup``, ``AuctionConfig``) are ``frozen=True`` — they are
  immutable inputs to the auction lifecycle.
* State objects (``ParticipantState``, ``AssignmentRecord``,
  ``AuctionState``) are mutable: the orchestrator evolves them as the
  auction progresses.
* Result objects (``RecordResult``, ``AlternativeSuggestion``,
  ``AuctionSummary``) are ``frozen=True`` — they are immutable outputs
  of single operations.

The :class:`Player` model is **reused** from :mod:`ml.optimizer.models` and
must not be redefined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Literal

from ml.mantra.roles import ALL_ROLES as _MANTRA_ALL_ROLES
from ml.optimizer.models import MANTRA_DEFAULT_QUOTAS, Player, Role, RulesetType

_MANTRA_ROLE_SET: Final[frozenset[str]] = frozenset(_MANTRA_ALL_ROLES)
_CLASSIC_ROLE_SET: Final[frozenset[str]] = frozenset({"P", "D", "C", "A"})


class ValuationMode(str, Enum):
    """Score metric used by VarEngine and the optimizer objective.

    PER_MATCH_RATING: rank by predicted per-match fantavoto (default, backward compatible).
    SEASON_VALUE: rank by season-total expected fanta-points (rating × predicted appearances).
    """

    PER_MATCH_RATING = "PER_MATCH_RATING"
    SEASON_VALUE = "SEASON_VALUE"


__all__ = [
    "ADJACENT_TIERS",
    "ALL_TIERS",
    "AlternativeSuggestion",
    "AlternativesConfig",
    "AssignmentRecord",
    "AuctionConfig",
    "AuctionState",
    "AuctionSummary",
    "MarketDriftConfig",
    "ParticipantSetup",
    "ParticipantState",
    "RecordResult",
    "Role",
    "RulesetType",
    "Tier",
    "ValuationMode",
]

# ---------------------------------------------------------------------------
# Tier taxonomy
# ---------------------------------------------------------------------------

Tier = Literal["LOW", "MID", "TOP"]
"""Percentile-based tier within a role for the price drift model."""

ALL_TIERS: Final[tuple[Tier, ...]] = ("LOW", "MID", "TOP")
"""Stable ordering used when iterating over tiers."""

ADJACENT_TIERS: Final[dict[Tier, tuple[Tier, ...]]] = {
    "LOW": ("MID",),
    "MID": ("LOW", "TOP"),
    "TOP": ("MID",),
}
"""Adjacent tiers that receive attenuated spillover from an update.

The order in each tuple is implementation detail; the membership defines
which tiers are eligible for the spillover step.  ``LOW`` and ``TOP`` have
exactly one neighbour; ``MID`` has two.
"""


# ---------------------------------------------------------------------------
# Configuration (immutable inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDriftConfig:
    """Configurazione del modello EWMA di aggiustamento dinamico dei prezzi.

    Tutti i parametri sono configurabili; nessun valore di dominio è
    hardcoded altrove.
    """

    alpha: float = 0.3
    """Peso dell'osservazione più recente nell'aggiornamento EWMA."""

    spillover_adjacent_tier: float = 0.25
    """Coefficiente di spillover verso i tier adiacenti dello stesso ruolo."""

    spillover_cross_role: float = 0.0
    """Hook di spillover cross-ruolo (disattivato di default)."""

    min_index: float = 0.5
    """Limite inferiore del price index (clamp)."""

    max_index: float = 1.8
    """Limite superiore del price index (clamp)."""

    tier_thresholds: tuple[float, float] = (0.4, 0.8)
    """Soglie ``(low, top)`` di percentile per classificare LOW/MID/TOP."""

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                f"MarketDriftConfig.alpha must be in (0, 1], got {self.alpha}"
            )
        if self.spillover_adjacent_tier < 0.0:
            raise ValueError(
                "MarketDriftConfig.spillover_adjacent_tier must be >= 0, got "
                f"{self.spillover_adjacent_tier}"
            )
        if self.spillover_cross_role < 0.0:
            raise ValueError(
                "MarketDriftConfig.spillover_cross_role must be >= 0, got "
                f"{self.spillover_cross_role}"
            )
        if self.min_index <= 0.0:
            raise ValueError(
                f"MarketDriftConfig.min_index must be > 0, got {self.min_index}"
            )
        if self.max_index < self.min_index:
            raise ValueError(
                "MarketDriftConfig.max_index must be >= min_index, got "
                f"max={self.max_index}, min={self.min_index}"
            )
        if len(self.tier_thresholds) != 2:
            raise ValueError(
                "MarketDriftConfig.tier_thresholds must have exactly 2 elements"
            )
        low, top = self.tier_thresholds
        if not 0.0 <= low < top <= 1.0:
            raise ValueError(
                "MarketDriftConfig.tier_thresholds must satisfy "
                f"0 <= low < top <= 1, got low={low}, top={top}"
            )


@dataclass(frozen=True)
class AlternativesConfig:
    """Configurazione delle alternative suggerite in tempo reale."""

    low_cost_percentile: float = 0.4
    """Soglia di percentile del ``expected_price`` per filtrare low-cost."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_cost_percentile <= 1.0:
            raise ValueError(
                "AlternativesConfig.low_cost_percentile must be in [0,1], got "
                f"{self.low_cost_percentile}"
            )


@dataclass(frozen=True)
class ParticipantSetup:
    """Setup di un singolo partecipante."""

    participant_id: str
    display_name: str
    budget_initial: int

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("ParticipantSetup.participant_id must be non-empty")
        if not self.display_name:
            raise ValueError("ParticipantSetup.display_name must be non-empty")
        if self.budget_initial <= 0:
            raise ValueError(
                "ParticipantSetup.budget_initial must be > 0, got "
                f"{self.budget_initial}"
            )


@dataclass(frozen=True)
class AuctionConfig:
    """Configurazione completa di un'asta."""

    num_participants: int
    role_quotas: dict[str, int] = field(
        default_factory=lambda: {"P": 3, "D": 8, "C": 8, "A": 6}
    )
    """Quote per ruolo. Per ``ruleset="CLASSIC"`` deve contenere esattamente
    le chiavi P/D/C/A. Per ``ruleset="MANTRA"`` deve contenere chiavi valide
    tra i 12 ruoli Mantra (vedi :data:`ml.mantra.roles.ALL_ROLES`); se non
    esplicitamente passato insieme a ``ruleset="MANTRA"``, va valorizzato dal
    chiamante (es. con :data:`ml.optimizer.models.MANTRA_DEFAULT_QUOTAS`) —
    il default di questo campo resta le quote CLASSIC per non alterare il
    comportamento dei chiamanti esistenti."""

    ruleset: RulesetType = "CLASSIC"
    """Ruleset dell'asta: ``"CLASSIC"`` (4 ruoli, default) o ``"MANTRA"``
    (12 ruoli, multi-slot). Riusa :data:`ml.optimizer.models.RulesetType`
    per restare coerente con il modulo Optimizer, nessuna duplicazione."""

    market_drift_config: MarketDriftConfig = field(default_factory=MarketDriftConfig)
    alternatives_config: AlternativesConfig = field(default_factory=AlternativesConfig)
    use_inflation_baseline: bool = False
    """Se ``True``, ``baseline_cost`` incorpora l'inflazione statica
    dell'ottimizzatore rosa (``estimate_effective_cost``)."""
    inflation_config: object = None
    """Configurazione opzionale dell'inflazione (se ``use_inflation_baseline``).
    Mantenuto come ``object`` per non duplicare i vincoli di
    :class:`InflationConfig`; viene validato a runtime."""
    valuation_mode: str = "PER_MATCH_RATING"
    """Score metric for VAR ranking: PER_MATCH_RATING or SEASON_VALUE."""

    hybrid_blend: float = 0.0
    """WS3 #2: weight in [0, 1] of the fpIbrido (MANTRA-ibrido) signal in
    VarEngine scoring. 0 = disabled (default). Same shape as
    OptimizationConfig.hybrid_blend."""

    reference_budget: int = 300
    """Budget per squadra su cui il listino (``player.cost``) è tarato.

    Il file delle quotazioni importato è storicamente calibrato su un
    fantacalcio a 300 crediti/squadra. Quando l'asta reale usa un budget
    per squadra diverso, il ``baseline_cost`` viene riproporzionato di un
    fattore ``budget_initial / reference_budget`` per riflettere il diverso
    potere d'acquisto.  Deve essere ``> 0``.
    """
    budget_initial: int = 300
    """Budget per squadra configurato per l'asta corrente.

    È il valore che l'operatore sceglie in fase di setup (es. 500 per leghe
    con più crediti, 100 per leghe corte).  Insieme a ``reference_budget``
    determina il fattore di scala applicato al listino in
    :func:`ml.auction.price_drift.compute_baseline_cost`.  Deve essere
    ``> 0``.  Il default ``300`` mantiene il comportamento storico
    (fattore di scala = 1.0).
    """

    def __post_init__(self) -> None:
        if self.num_participants < 1:
            raise ValueError(
                "AuctionConfig.num_participants must be >= 1, got "
                f"{self.num_participants}"
            )
        if (
            self.ruleset == "MANTRA"
            and set(self.role_quotas.keys()) == _CLASSIC_ROLE_SET
        ):
            # Caller opted into MANTRA but left role_quotas at its CLASSIC
            # default (didn't pass one explicitly) — fall back to the shared
            # MANTRA default quotas, same convenience already provided by
            # OptimizationConfig.mantra_role_quotas in ml.optimizer.models.
            object.__setattr__(self, "role_quotas", dict(MANTRA_DEFAULT_QUOTAS))
        if self.ruleset == "CLASSIC":
            if set(self.role_quotas.keys()) != _CLASSIC_ROLE_SET:
                raise ValueError(
                    "AuctionConfig.role_quotas must include exactly P/D/C/A, got "
                    f"{sorted(self.role_quotas.keys())}"
                )
        elif self.ruleset == "MANTRA":
            unknown = set(self.role_quotas.keys()) - _MANTRA_ROLE_SET
            if unknown:
                raise ValueError(
                    "AuctionConfig.role_quotas contains roles not valid for "
                    f"ruleset=MANTRA: {sorted(unknown)}. Valid roles: "
                    f"{sorted(_MANTRA_ROLE_SET)}"
                )
            if not self.role_quotas:
                raise ValueError(
                    "AuctionConfig.role_quotas must be non-empty for ruleset=MANTRA"
                )
        else:
            raise ValueError(
                f"AuctionConfig.ruleset must be CLASSIC or MANTRA, got {self.ruleset!r}"
            )
        for role, q in self.role_quotas.items():
            if q <= 0:
                raise ValueError(
                    f"AuctionConfig.role_quotas[{role}] must be > 0, got {q}"
                )
        if self.use_inflation_baseline and self.inflation_config is None:
            raise ValueError(
                "AuctionConfig.inflation_config is required when "
                "use_inflation_baseline=True"
            )
        if self.reference_budget <= 0:
            raise ValueError(
                "AuctionConfig.reference_budget must be > 0, got "
                f"{self.reference_budget}"
            )
        if self.budget_initial <= 0:
            raise ValueError(
                f"AuctionConfig.budget_initial must be > 0, got {self.budget_initial}"
            )
        if not 0.0 <= self.hybrid_blend <= 1.0:
            raise ValueError(
                f"AuctionConfig.hybrid_blend must be in [0, 1], got {self.hybrid_blend}"
            )


# ---------------------------------------------------------------------------
# Mutable state
# ---------------------------------------------------------------------------


@dataclass
class ParticipantState:
    """Stato corrente di un singolo partecipante."""

    participant_id: str
    display_name: str
    budget_residual: int
    squad: list[Player]
    role_breakdown: dict[Role, int]


@dataclass
class AssignmentRecord:
    """Record di un'assegnazione registrata, in ordine cronologico.

    ``price_index_snapshot_before`` è una copia profonda del ``price_index``
    immediatamente prima dell'aggiornamento EWMA: serve a
    :func:`undo_last_assignment` per ripristinare l'indice in modo
    deterministico senza dover invertire l'operazione.

    ``assigned_slot`` è lo slot di ruolo effettivamente occupato nella rosa
    del vincitore. In modalità CLASSIC coincide sempre con ``role``
    (``player.role``). In modalità MANTRA è uno dei codici in
    ``player.eligible_roles`` (es. ``"Dd"``, ``"E"``) e può differire dal
    ``player.role`` classico — analogo alla variabile ``x_ir`` del solver ILP
    ("questo giocatore occupa lo slot Trq/Dc/Br/... anche se eleggibile per
    più ruoli"). Usato da ``role_breakdown``, undo e serializzazione.
    """

    sequence_number: int
    player: Player
    winner_participant_id: str
    final_price: int
    role: Role
    tier: Tier
    price_index_before: float
    price_index_after: float
    price_index_snapshot_before: dict[Role, dict[Tier, float]] = field(
        default_factory=dict
    )
    assigned_slot: str | None = None
    """Slot di ruolo effettivamente riempito. ``None`` solo per record
    legacy deserializzati pre-Fase-3; in quel caso i consumer devono
    trattarlo come ``role``. I nuovi record lo popolano sempre."""


@dataclass
class AuctionState:
    """Stato evolutivo di un'asta live.

    ``role_percentile_map`` è una cache derivata (``player_id -> percentile``)
    calcolata una sola volta al bootstrap dal pool completo.  Dipende solo
    dalla composizione iniziale del pool (i giocatori non cambiano ruolo
    durante l'asta), quindi è parte coerente dello stato serializzabile.
    """

    config: AuctionConfig
    participants: dict[str, ParticipantState]
    assignments: list[AssignmentRecord]
    price_index: dict[Role, dict[Tier, float]]
    available_pool: list[Player]
    role_percentile_map: dict[str, float] = field(default_factory=dict)
    team_strength_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output models (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordResult:
    """Esito di un tentativo di registrazione di un'assegnazione."""

    success: bool
    updated_state: AuctionState | None = None
    rejection_reason: str | None = None
    rejection_code: str | None = None
    """Codice macchina del rifiuto (per telemetria/log), es. ``"role_full"``."""


@dataclass(frozen=True)
class AlternativeSuggestion:
    """Suggerimento di alternative per un giocatore target.

    ``low_cost_alternative`` / ``closest_alternative`` restano le due
    euristiche classiche (G2 parity).  WS3 estende con:

    * ``diversified_alternatives`` — mini-fronte Pareto su (score, -price,
      value ratio), deduplicato rispetto alle due euristiche fisse.
    * ``max_affordable_bid`` — prezzo massimo pagabile dal partecipante
      indicato rispettando la riserva crediti (WS3 #4 sensitivity).
    * ``strategy_price_cap`` — soglia strategia-aware (WS3 #5), se una
      ``StrategyProfile`` è stata fornita al chiamante.
    """

    target_player_id: str
    low_cost_alternative: Player | None
    closest_alternative: Player | None
    reason_if_none: str | None = None
    diversified_alternatives: tuple = ()
    """Up to N non-dominated candidates (Player instances)."""
    max_affordable_bid: int | None = None
    strategy_price_cap: int | None = None


@dataclass(frozen=True)
class AuctionSummary:
    """Riepilogo corrente dell'asta."""

    participants: list[ParticipantState]
    assignments: list[AssignmentRecord]
    price_index: dict[Role, dict[Tier, float]]
    completion_probability: dict[str, float] | None = None
    """WS3 #1: map participant_id → P(complete roster | residual budget)."""
