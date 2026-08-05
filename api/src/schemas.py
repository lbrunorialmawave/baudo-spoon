from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

T = TypeVar("T")

# ── Shared base for camelCase JSON serialisation ──────────────────────────────


class _CamelModel(BaseModel):
    """Base class that emits camelCase keys in JSON responses."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class LeagueSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    comp_id: str
    slug: str


class SeasonSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    season_start: int
    season_label: str
    scraped_at: Optional[datetime] = None
    league: LeagueSchema


class MatchStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_date: Optional[str] = None
    round_num: Optional[int] = None
    match_name: str
    score: Optional[str] = None
    status: Optional[str] = None
    url: Optional[str] = None
    team: str
    side: Optional[str] = None
    opponent: Optional[str] = None
    goals_scored: Optional[int] = None
    goals_conceded: Optional[int] = None
    points: Optional[int] = None
    stats: dict[str, Any]
    ingested_at: datetime
    season: SeasonSchema


class PlayerSeasonStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fotmob_season_id: int
    stat_category: str
    rank: Optional[int] = None
    player_fotmob_id: int
    player_name: str
    team_fotmob_id: Optional[int] = None
    team_name: Optional[str] = None
    value: Optional[Decimal] = None
    ingested_at: datetime
    season: SeasonSchema


class TeamSeasonStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fotmob_season_id: int
    stat_category: str
    rank: Optional[int] = None
    team_fotmob_id: int
    team_name: str
    value: Optional[Decimal] = None
    ingested_at: datetime
    season: SeasonSchema


class PaginatedResponse(BaseModel, Generic[T]):
    total: int
    page: int
    size: int
    items: list[T]


# ── ML / Intelligence schemas ─────────────────────────────────────────────────


class PlayerPredictionSchema(_CamelModel):
    """Single player prediction record from the ML artifact."""

    player_name: str
    player_fotmob_id: Optional[int] = None
    team_name: Optional[str] = None
    canonical_role: Optional[str] = None
    season: Optional[str] = None
    fantavoto_medio: Optional[float] = None  # actual (when available)
    predicted: float
    # Phase 3+ enrichments (optional — absent in older artifacts)
    confidence: Optional[float] = None
    prediction_interval_low: Optional[float] = None
    prediction_interval_high: Optional[float] = None
    expected_minutes: Optional[float] = None


class PlayerVarSchema(_CamelModel):
    """Value Above Replacement record for a single player."""

    player_id: str
    player_name: Optional[str] = None
    role: str
    projected_score: float
    season_value: float | None = None
    start_probability: float | None = None
    replacement_level_score: float
    var_score: float
    expected_price: float
    esv: float
    calibrated: bool


class VarResultsResponse(_CamelModel):
    """Response for GET /intelligence/var/players."""

    run_id: str
    calibrated: bool
    total: int
    items: list[PlayerVarSchema]


class NextSeasonPredictionSchema(_CamelModel):
    player_name: str
    player_fotmob_id: Optional[int] = None
    predicted_next_fantavoto: float


class ModelComparisonSchema(_CamelModel):
    model: str
    rmse: float
    mae: float
    r2: float


class PredictionsResponse(_CamelModel):
    run_id: str
    best_model: str
    role_partitioned: bool
    predictions: list[PlayerPredictionSchema]
    model_comparison: list[ModelComparisonSchema]
    next_season_predictions: list[NextSeasonPredictionSchema]


class PlayerClusterSchema(_CamelModel):
    """Cluster membership for a single player."""

    player_name: str
    player_fotmob_id: Optional[int] = None
    team_name: Optional[str] = None
    canonical_role: Optional[str] = None
    cluster_id: int
    pca_0: Optional[float] = None
    pca_1: Optional[float] = None
    predicted_fantavoto: Optional[float] = None


class LowCostAlternativeSchema(_CamelModel):
    """Low-cost clone recommendation for a top-percentile player.

    Field names mirror the LowCostAlternative dataclass produced by
    ml/clustering/kmeans.py so that dataclasses.asdict() output maps directly.
    """

    top_player_id: Optional[int] = None
    top_player_name: str
    top_player_team: Optional[str] = None
    top_player_fantavoto: Optional[float] = None
    alt_player_id: Optional[int] = None
    alt_player_name: str
    alt_player_team: Optional[str] = None
    alt_player_fantavoto: Optional[float] = None
    cluster_id: int
    distance: float


class ClusteringStatsSchema(_CamelModel):
    n_clusters: int
    silhouette: Optional[float] = None
    inertia: Optional[float] = None
    pca_explained_variance: Optional[list[float]] = None


class AlternativesResponse(_CamelModel):
    clustering_stats: ClusteringStatsSchema
    player_clusters: list[PlayerClusterSchema]
    low_cost_recommendations: list[LowCostAlternativeSchema]


# ── Quotation & ID-mapping schemas ────────────────────────────────────────────


class PlayerQuotationSchema(_CamelModel):
    """One (fantacalcio_id, season_start) row from ``player_quotations``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    fantacalcio_id: int
    season_start: int
    role: str
    team: str
    player_name: str
    qt_a: int
    qt_i: int
    diff_val: int
    qt_a_m: Optional[int] = None
    qt_i_m: Optional[int] = None
    diff_val_m: Optional[int] = None
    fvm: Optional[int] = None
    fvm_m: Optional[int] = None
    source: str
    imported_at: datetime


class PlayerQuotationWithMappingSchema(PlayerQuotationSchema):
    """Quotation joined to its id-map row (left-join: mapping may be null)."""

    player_fotmob_id: Optional[int] = None
    name_fotmob: Optional[str] = None
    team_fotmob: Optional[str] = None
    match_method: Optional[str] = None
    confidence: Optional[float] = None
    ruolo_primario: Optional[str] = None
    ruoli_mantra: Optional[list[str]] = None


class PlayerIdMapSchema(_CamelModel):
    """One (fantacalcio_id, season_start) row from ``player_id_map``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    fantacalcio_id: int
    season_start: int
    player_fotmob_id: Optional[int] = None
    name_fantacalcio: str
    name_fotmob: Optional[str] = None
    team_fantacalcio: Optional[str] = None
    team_fotmob: Optional[str] = None
    canonical_role: Optional[str] = None
    match_method: str
    confidence: float
    resolved_from_history: bool = False
    created_at: datetime
    updated_at: datetime
    # MANTRA 12-role fields (from player_mantra_roles, may be null)
    ruoli_mantra: Optional[list[str]] = None
    ruolo_primario: Optional[str] = None


class UpdateIdMappingRequest(_CamelModel):
    """Request body to manually update a Fantacalcio ↔ FotMob mapping.

    All fields are optional except those identifying the row; only
    non-``None`` fields will be updated.
    """

    player_fotmob_id: Optional[int] = None
    """FotMob player ID to assign. Set to ``-1`` to clear/keep unmatched."""
    name_fotmob: Optional[str] = None
    """FotMob player name (informational)."""
    team_fotmob: Optional[str] = None
    """FotMob team name (optional override)."""
    canonical_role: Optional[str] = None
    """Override canonical role (GK/DEF/MID/FWD)."""
    note: Optional[str] = None
    """Free-text note about this override."""
    # MANTRA role overrides (optional)
    ruoli_mantra: Optional[list[str]] = None
    """Override MANTRA roles (e.g. ["Dd", "E"])."""
    ruolo_primario: Optional[str] = None
    """Override primary MANTRA role."""
    data_validated: Optional[bool] = None
    """Mark as validated by the user."""


class QuotationRoleAggregateSchema(_CamelModel):
    """One row of ``GET /quotations/stats``: aggregate per role+season."""

    season_start: int
    role: str
    n_players: int
    avg_qt_a: float
    avg_qt_i: float
    median_qt_a: float
    min_qt_a: int
    max_qt_a: int
    avg_fvm: Optional[float] = None


class QuotationStatsResponse(_CamelModel):
    """Response of ``GET /quotations/stats``."""

    total_quotations: int
    seasons: list[int]
    by_season_role: list[QuotationRoleAggregateSchema]
    n_teams: int
    coverage: dict[str, int]  # mapping method → row count


class ManualResolutionSchema(_CamelModel):
    """A single manual resolution record."""

    id: int
    fantacalcio_id: int
    player_fotmob_id: int
    season_start: int
    name_fantacalcio: str
    team_fantacalcio: Optional[str] = None
    canonical_role: Optional[str] = None
    name_fotmob: Optional[str] = None
    team_fotmob: Optional[str] = None
    resolved_by: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime


class ManualResolutionStatsResponse(_CamelModel):
    """Response of ``GET /intelligence/id-mapping/resolutions/stats``."""

    total: int
    unique_players: int
    by_season: dict[str, int]


class IdMappingStatsResponse(_CamelModel):
    """Response of ``GET /intelligence/id-mapping/stats``."""

    total: int
    matched: int
    unmatched: int
    match_rate: float
    by_season: dict[str, dict[str, int]]  # season → {method: count}
    by_method: dict[str, int]


# ── Optimizer schemas ────────────────────────────────────────────────────────


class FormationSchema(_CamelModel):
    """Modulo tattico: numero di difensori, centrocampisti, attaccanti titolari."""

    label: str
    defenders: int
    midfielders: int
    forwards: int


class InflationConfigSchema(_CamelModel):
    """Configurazione della funzione di inflazione del costo d'asta."""

    inflation_percentile_threshold: float = 0.7
    max_inflation_multiplier: float = 1.6
    base_inflation_rate: float = 0.05
    baseline_participants: int = 8
    team_strength_multiplier: float = Field(default=0.0, ge=0.0)
    """Peso dell'aggiustamento team-strength (Elo) sul costo effettivo.

    Quando ``> 0``, il costo effettivo dei giocatori appartenenti a
    squadre forti viene moltiplicato per ``1 + weight * normalized_elo``,
    dove ``normalized_elo`` è il punteggio Elo normalizzato in ``[0, 1]``
    caricato da :func:`ml.optimizer.team_strength.load_team_strength_scores`.
    Default ``0.0`` preserva il comportamento storico (nessun boost).
    """


class PlayerSchema(_CamelModel):
    """Singolo giocatore nel pool di ottimizzazione."""

    player_id: str
    name: str
    role: str  # P | D | C | A
    real_team: str
    cost: int
    projected_score: float
    reliability_weight: Optional[float] = None
    eligible_roles: list[str] = Field(default_factory=list)  # MANTRA only
    prediction_std: Optional[float] = None  # ensemble std; drives risk_aversion penalty
    historical_overpay_ratio: Optional[float] = None  # Picco/listino from pilastro4
    season_value: Optional[float] = None
    start_probability: Optional[float] = None


class OptimizationRequest(_CamelModel):
    """Request body per gli endpoint di ottimizzazione rosa.

    Tutti i campi sono opzionali tranne ``season_start``: l'API applica
    default coerenti con la specifica del modulo optimizer. ``pool_override``
    consente di bypassare il fetch dal DB+ML artifact (utile per test o
    per pool custom costruiti lato client).
    """

    season_start: int
    budget: int = Field(default=500, gt=0)
    num_participants: int = Field(default=8, ge=2, le=20)
    min_qt_a: int = 1
    min_distinct_teams: int = 12
    max_players_per_team: int = 4
    big_teams: list[str] = [
        "Inter",
        "Milan",
        "Juventus",
        "Napoli",
    ]
    big_teams_cap: int = 10
    formations: list[FormationSchema] = Field(
        default=[
            FormationSchema(label="3-4-3", defenders=3, midfielders=4, forwards=3),
            FormationSchema(label="4-3-3", defenders=4, midfielders=3, forwards=3),
            FormationSchema(label="4-4-2", defenders=4, midfielders=4, forwards=2),
            FormationSchema(label="3-5-2", defenders=3, midfielders=5, forwards=2),
        ],
        max_length=6,
    )
    inflation_config: InflationConfigSchema = InflationConfigSchema()
    solver_timeout_seconds: int = Field(default=30, ge=1, le=60)
    max_single_player_budget_share: float = Field(default=0.30, gt=0.0, le=1.0)
    must_include: list[str] = Field(default_factory=list, max_length=25)
    exclude: list[str] = Field(default_factory=list, max_length=200)
    ruleset: str = "CLASSIC"  # "CLASSIC" | "MANTRA"
    mantra_role_quotas: Optional[dict[str, int]] = None
    # When set, the squad is guaranteed to be able to field this module.
    # All formations in `formations` are still evaluated post-hoc and reported
    # in formation_feasibility, but only this one is a hard solver constraint.
    preferred_formation: Optional[FormationSchema] = None
    risk_aversion: float = Field(default=0.0, ge=0.0)
    var_blend: float = Field(default=0.0, ge=0.0, le=1.0)
    esv_weight: float = Field(default=0.0, ge=0.0)
    # Pool pre-filter: esclude i giocatori con start_probability inferiore
    # alla soglia PRIMA della costruzione della pool passata al solver ILP.
    # `None` (default) ⇒ nessun filtro (compatibilità con richieste legacy).
    # Quando `var_blend > 0` oppure `esv_weight > 0`, lo stesso valore viene
    # propagato al VarEngine per garantire coerenza tra ranking VAR e pool.
    min_start_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Soglia minima di probabilità di titolarità (0..1) per "
            "includere un giocatore nella pool del solver. I giocatori "
            "con start_probability < soglia vengono filtrati PRIMA "
            "dell'ILP, analogamente a quanto fa AuctionConfigSchema. "
            "`None` = nessun filtro (default)."
        ),
    )
    # Replacement level per il calcolo VAR/ESV quando il blend è attivo
    # (`var_blend > 0` oppure `esv_weight > 0`). Valori ammessi:
    # "percentile" (default, bottom-N% per ruolo) oppure
    # "roster_depth" (quota di rosa per ruolo). Allineato al campo
    # omonimo di AuctionConfigSchema.
    replacement_method: str = Field(
        default="percentile",
        description=(
            "Metodo di calcolo del replacement level per il VAR/ESV "
            "blend. Identico al campo omonimo di AuctionConfigSchema: "
            "'percentile' (bottom-N% per ruolo) oppure 'roster_depth' "
            "(quota di rosa per ruolo)."
        ),
    )
    valuation_mode: str = Field(
        default="PER_MATCH_RATING",
        description="PER_MATCH_RATING (default) or SEASON_VALUE (needs season_value on pool).",
    )
    strategy_names: Optional[list[str]] = None
    custom_strategies: Optional[list["StrategyProfileSchema"]] = None
    pool_override: Optional[list[PlayerSchema]] = Field(default=None, max_length=500)
    monte_carlo: Optional["MonteCarloConfigSchema"] = Field(default=None)
    diversify_strategies: bool = Field(
        default=False,
        description="If true on /multi, re-solve secondary strategies excluding core of the primary when overlap is high.",
    )
    near_optimal: Optional["NearOptimalConfigSchema"] = Field(default=None)


class MonteCarloConfigSchema(_CamelModel):
    """Monte Carlo robustness block. Default off — omit for legacy deterministic ILP.

    Trade-offs:
    * ``mean_std``: 1× ILP latency, risk-adjusted point estimate (mean − λ·std).
    * ``saa_frequency``: N× ILP latency; returns selection frequency + stability_index.
      Prefer N≤50 sync; N>200 → POST /optimize/jobs. Caps: API_OPTIMIZER_MAX_SIMULATIONS.
    * Do not combine high ``risk_aversion`` with MC without reading both effects
      (risk_aversion shrinks projected_score once; MC re-samples scores).
    """
    enabled: bool = Field(default=False, description="Master switch; false keeps deterministic path.")
    n_simulations: int = Field(
        default=200, ge=1, le=1000,
        description="SAA scenarios. Sync path capped by API_OPTIMIZER_MAX_SIMULATIONS.",
    )
    mode: str = Field(
        default="saa_frequency",
        description="mean_std (fast risk-adjusted single solve) | saa_frequency (distributional).",
    )
    risk_lambda: float = Field(default=0.5, ge=0.0, description="Only for mean_std: mean − λ·std.")
    min_selection_frequency: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Warn when representative players fall below this SAA frequency.",
    )
    random_seed: int = Field(default=42, description="Reproducible residual draws / scenario order.")
    timeout_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Soft SAA wall budget; 0 → API_OPTIMIZER_SAA_TIMEOUT_SECONDS default.",
    )


class MonteCarloSummarySchema(_CamelModel):
    n_simulations: int
    mode: str
    random_seed: int = 42
    stability_index: float = 0.0
    selection_frequency: dict[str, float] = Field(default_factory=dict)
    squad_score_percentiles: dict[str, float] = Field(default_factory=dict)
    mean_pairwise_jaccard: float = 0.0
    scenarios_completed: int = 0
    wall_time_seconds: float = 0.0
    sampling_methods_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    # Residual provenance (walk-forward file vs prediction_std fallback)
    residual_source: Optional[str] = None
    residual_using: Optional[str] = None
    residual_rows: Optional[int] = None
    residual_merged_rows: Optional[int] = None


class NearOptimalConfigSchema(_CamelModel):
    enabled: bool = False
    n_alternatives: int = Field(default=3, ge=1, le=10)
    exclude_top_m: int = Field(default=2, ge=1, le=10)
    max_score_drop_pct: float = Field(default=0.15, ge=0.0, le=1.0)


class NearOptimalAlternativeSchema(_CamelModel):
    excluded_player_ids: list[str]
    score_delta: float
    score_delta_pct: float
    squad: list["SquadPlayerSchema"]
    total_projected_score: float
    status: str


class DiversityMetricsSchema(_CamelModel):
    mean_pairwise_jaccard: float = 0.0
    max_pairwise_jaccard: float = 0.0
    min_pairwise_jaccard: float = 0.0
    mean_overlap_count: float = 0.0
    max_overlap_count: int = 0
    low_diversity: bool = False
    pairwise_jaccard: dict[str, float] = Field(default_factory=dict)


class OptimizeJobCreateResponse(_CamelModel):
    job_id: str
    status: str = "queued"


class OptimizeJobStatusSchema(_CamelModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None
    result: Optional["OptimizationResultSchema"] = None
    monte_carlo_summary: Optional[MonteCarloSummarySchema] = None


class SquadPlayerSchema(_CamelModel):
    """Giocatore come appare nella rosa selezionata."""

    player_id: str
    name: str
    role: str
    real_team: str
    cost: int
    projected_score: float
    effective_cost: float


class OptimizationResultSchema(_CamelModel):
    """Output di una singola strategia di ottimizzazione."""

    strategy_name: str
    status: str
    squad: list[SquadPlayerSchema]
    total_nominal_cost: int
    total_effective_cost: float
    total_projected_score: float
    budget_residual: float
    role_breakdown: dict[str, int]
    team_breakdown: dict[str, int]
    distinct_teams_count: int
    big_teams_players_count: int
    formation_feasibility: dict[str, bool]
    diagnostics: dict[str, Any]
    win_probability: Optional[float] = None
    monte_carlo_summary: Optional[MonteCarloSummarySchema] = None
    near_optimal: list[NearOptimalAlternativeSchema] = Field(default_factory=list)


class MultiStrategyResultSchema(_CamelModel):
    """Output di ``POST /optimize/multi``: una entry per strategia richiesta."""

    results: dict[str, OptimizationResultSchema]
    monte_carlo_summary: Optional[MonteCarloSummarySchema] = None
    diversity: Optional[DiversityMetricsSchema] = None


class StrategyProfileSchema(_CamelModel):
    """Profilo di strategia di ottimizzazione (read-only, informativo)."""

    name: str
    role_weight: dict[str, float]
    min_budget_share_by_roles: Optional[tuple[list[str], float]] = None
    max_top_tier_players: Optional[int] = None
    top_tier_cost_threshold: Optional[float] = None


class DefaultStrategiesResponse(_CamelModel):
    """Response di ``GET /optimize/strategies``: lista strategie di default."""

    strategies: list[StrategyProfileSchema]


# ── Auction (live auction tracker) schemas ──────────────────────────────────


class MarketDriftConfigSchema(_CamelModel):
    """Configurazione EWMA per il price drift."""

    alpha: float = 0.3
    spillover_adjacent_tier: float = 0.25
    spillover_cross_role: float = 0.0
    min_index: float = 0.5
    max_index: float = 1.8
    tier_thresholds: list[float] = [0.4, 0.8]


class AlternativesConfigSchema(_CamelModel):
    """Configurazione del modulo di suggerimento alternative."""

    low_cost_percentile: float = 0.4


class AuctionParticipantSetupSchema(_CamelModel):
    """Setup di un singolo partecipante all'asta."""

    participant_id: str
    display_name: str
    budget_initial: int


class AuctionConfigSchema(_CamelModel):
    """Configurazione completa di un'asta live."""

    num_participants: int
    role_quotas: dict[str, int] = {"P": 3, "D": 8, "C": 8, "A": 6}
    market_drift_config: MarketDriftConfigSchema = MarketDriftConfigSchema()
    alternatives_config: AlternativesConfigSchema = AlternativesConfigSchema()
    use_inflation_baseline: bool = False
    inflation_config: Optional[InflationConfigSchema] = None
    """Custom inflation parameters. When None and use_inflation_baseline=True,
    server-side defaults are used."""
    reference_budget: int = 300
    """Budget per squadra su cui il listino è tarato (default: 300 cr).

    Il file delle quotazioni importato è storicamente calibrato su un
    fantacalcio a 300 crediti/squadra.  Quando l'asta reale usa un budget
    per squadra diverso (es. 500 cr), il ``baseline_cost`` usato dal
    price drift EWMA viene riproporzionato per il fattore
    ``budgetInitial / referenceBudget`` in modo che le aspettative di
    prezzo partano da un valore coerente con il potere d'acquisto reale
    della lega.  Deve essere ``> 0``.
    """
    budget_initial: int = 300
    """Budget per squadra configurato per l'asta corrente (default: 300 cr).

    È il valore che l'operatore sceglie in fase di setup (es. 500 per
    leghe con più crediti, 100 per leghe corte).  Insieme a
    ``referenceBudget`` determina il fattore di scala applicato al
    listino nell'orchestratore dell'asta.  Deve essere ``> 0``.  Il
    default ``300`` mantiene il comportamento storico (fattore 1.0).
    """

    valuation_mode: str = "PER_MATCH_RATING"
    """Score metric for VAR ranking: PER_MATCH_RATING (default) or SEASON_VALUE.

    When SEASON_VALUE, the VarEngine ranks players by season-total expected
    fanta-points (rating × predicted appearances) instead of raw per-match rating.
    """

    min_start_probability: float | None = None
    """Minimum start_probability to include a player in the VAR ranking.

    Players below this threshold are excluded from the ranked output but remain
    in the full pool. None (default) = no filtering.
    """

    replacement_method: str = "percentile"
    """How to compute replacement level: 'percentile' (bottom 10th pctile of pool,
    default) or 'roster_depth' (score at num_participants × role_quota rank).
    """

    @field_validator("reference_budget", "budget_initial")
    @classmethod
    def _validate_positive_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v


class AuctionPlayerSchema(_CamelModel):
    """Player passato al backend per inizializzare il pool d'asta."""

    player_id: str
    name: str
    role: str  # P | D | C | A
    real_team: str
    cost: int
    projected_score: float
    season_value: float | None = None
    start_probability: float | None = None


class InitializeAuctionRequest(_CamelModel):
    """Request body per ``POST /auction/init``.

    Fornisce i partecipanti, la configurazione di mercato, e (opzionalmente)
    il pool di giocatori disponibili per l'asta.

    * ``season_start`` è **obbligatorio**: serve come chiave di lookup per
      le predizioni ML e per le quotazioni correnti nel DB.
    * ``player_pool`` è **opzionale**: se omesso, il backend costruisce
      il pool chiamando :meth:`DataRepository.get_player_pool` con
      ``min_qt_a=1`` (tutti i giocatori con quotazione disponibile),
      replicando il pattern di ``OptimizationRequest.pool_override``.
      Passarlo esplicito resta utile per test, per fixture custom, o
      per sessioni che partono da un pool ristretto (es. solo i
      giocatori rimasti dopo un'asta precedente).
    """

    season_start: int
    participants: list[AuctionParticipantSetupSchema]
    config: AuctionConfigSchema
    player_pool: Optional[list[AuctionPlayerSchema]] = None


class InitializeAuctionResponse(_CamelModel):
    """Response di ``POST /auction/init``: id di sessione generato lato server."""

    session_id: str


class RecordAssignmentRequest(_CamelModel):
    """Request body per ``POST /auction/{session_id}/record``."""

    player_id: str
    winner_participant_id: str
    final_price: int


class RecordAssignmentResponse(_CamelModel):
    """Response di ``POST /auction/{session_id}/record``.

    Su success, include uno snapshot minimo (sequence_number,
    price_index_after); su rifiuto, popolati ``rejection_code`` e
    ``rejection_reason``.
    """

    success: bool
    sequence_number: Optional[int] = None
    price_index_after: Optional[float] = None
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None


class ProjectionResponse(_CamelModel):
    """Response di ``GET /auction/{session_id}/projection/{player_id}``."""

    player_id: str
    expected_price: float
    tier: str  # LOW | MID | TOP


class AlternativesRequest(_CamelModel):
    """Request body opzionale per ``GET /auction/{session_id}/alternatives``."""

    config: Optional[AlternativesConfigSchema] = None


class AuctionPlayerSummarySchema(_CamelModel):
    """Player serializzato nelle risposte di summary."""

    player_id: str
    name: str
    real_team: str
    role: str
    cost: int
    projected_score: float
    season_value: Optional[float] = None
    start_probability: Optional[float] = None


class AuctionParticipantStateSchema(_CamelModel):
    """Stato corrente di un partecipante serializzato."""

    participant_id: str
    display_name: str
    budget_residual: int
    squad: list[AuctionPlayerSummarySchema]
    role_breakdown: dict[str, int]


class AssignmentRecordSchema(_CamelModel):
    """Record di un'assegnazione serializzato."""

    sequence_number: int
    player: AuctionPlayerSummarySchema
    winner_participant_id: str
    final_price: int
    role: str
    tier: str
    price_index_before: float
    price_index_after: float


class AuctionSummarySchema(_CamelModel):
    """Response di ``GET /auction/{session_id}/summary``."""

    participants: list[AuctionParticipantStateSchema]
    assignments: list[AssignmentRecordSchema]
    price_index: dict[str, dict[str, float]]


class AlternativesResponse(_CamelModel):
    """Response di ``GET /auction/{session_id}/alternatives/{player_id}``."""

    target_player_id: str
    low_cost_alternative: Optional[AuctionPlayerSummarySchema] = None
    closest_alternative: Optional[AuctionPlayerSummarySchema] = None
    reason_if_none: Optional[str] = None


class SerializedAuctionStateResponse(_CamelModel):
    """Response di ``GET /auction/{session_id}/serialize``."""

    payload: dict[str, object]


class VarRankingItemSchema(_CamelModel):
    """Single player entry in the VAR/ESV ranking."""

    player_id: str
    name: str
    role: str
    projected_score: float
    var_score: float
    expected_price: float
    esv: float
    calibrated: bool
    buy_signal: bool  # esv > 0
    season_value: Optional[float] = None
    start_probability: Optional[float] = None


class VarRankingResponse(_CamelModel):
    """Response of ``GET /auction/{session_id}/var-ranking``."""

    session_id: str
    items: list[VarRankingItemSchema]
    using_live_prices: bool
