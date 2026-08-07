from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, TypeVar

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
    scraped_at: datetime | None = None
    league: LeagueSchema


class MatchStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_date: str | None = None
    round_num: int | None = None
    match_name: str
    score: str | None = None
    status: str | None = None
    url: str | None = None
    team: str
    side: str | None = None
    opponent: str | None = None
    goals_scored: int | None = None
    goals_conceded: int | None = None
    points: int | None = None
    stats: dict[str, Any]
    ingested_at: datetime
    season: SeasonSchema


class PlayerSeasonStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fotmob_season_id: int
    stat_category: str
    rank: int | None = None
    player_fotmob_id: int
    player_name: str
    team_fotmob_id: int | None = None
    team_name: str | None = None
    value: Decimal | None = None
    ingested_at: datetime
    season: SeasonSchema


class TeamSeasonStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fotmob_season_id: int
    stat_category: str
    rank: int | None = None
    team_fotmob_id: int
    team_name: str
    value: Decimal | None = None
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
    player_fotmob_id: int | None = None
    team_name: str | None = None
    canonical_role: str | None = None
    season: str | None = None
    fantavoto_medio: float | None = None  # actual (when available)
    predicted: float
    # Phase 3+ enrichments (optional — absent in older artifacts)
    confidence: float | None = None
    prediction_interval_low: float | None = None
    prediction_interval_high: float | None = None
    expected_minutes: float | None = None


class PlayerVarSchema(_CamelModel):
    """Value Above Replacement record for a single player."""

    player_id: str
    player_name: str | None = None
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
    player_fotmob_id: int | None = None
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
    player_fotmob_id: int | None = None
    team_name: str | None = None
    canonical_role: str | None = None
    cluster_id: int
    pca_0: float | None = None
    pca_1: float | None = None
    predicted_fantavoto: float | None = None


class LowCostAlternativeSchema(_CamelModel):
    """Low-cost clone recommendation for a top-percentile player.

    Field names mirror the LowCostAlternative dataclass produced by
    ml/clustering/kmeans.py so that dataclasses.asdict() output maps directly.
    """

    top_player_id: int | None = None
    top_player_name: str
    top_player_team: str | None = None
    top_player_fantavoto: float | None = None
    alt_player_id: int | None = None
    alt_player_name: str
    alt_player_team: str | None = None
    alt_player_fantavoto: float | None = None
    cluster_id: int
    distance: float


class ClusteringStatsSchema(_CamelModel):
    n_clusters: int
    silhouette: float | None = None
    inertia: float | None = None
    pca_explained_variance: list[float] | None = None


class LowCostAlternativesResponse(_CamelModel):
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
    qt_a_m: int | None = None
    qt_i_m: int | None = None
    diff_val_m: int | None = None
    fvm: int | None = None
    fvm_m: int | None = None
    source: str
    imported_at: datetime


class PlayerQuotationWithMappingSchema(PlayerQuotationSchema):
    """Quotation joined to its id-map row (left-join: mapping may be null)."""

    player_fotmob_id: int | None = None
    name_fotmob: str | None = None
    team_fotmob: str | None = None
    match_method: str | None = None
    confidence: float | None = None
    ruolo_primario: str | None = None
    ruoli_mantra: list[str] | None = None


class PlayerIdMapSchema(_CamelModel):
    """One (fantacalcio_id, season_start) row from ``player_id_map``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    fantacalcio_id: int
    season_start: int
    player_fotmob_id: int | None = None
    name_fantacalcio: str
    name_fotmob: str | None = None
    team_fantacalcio: str | None = None
    team_fotmob: str | None = None
    canonical_role: str | None = None
    match_method: str
    confidence: float
    resolved_from_history: bool = False
    created_at: datetime
    updated_at: datetime
    # MANTRA 12-role fields (from player_mantra_roles, may be null)
    ruoli_mantra: list[str] | None = None
    ruolo_primario: str | None = None


class UpdateIdMappingRequest(_CamelModel):
    """Request body to manually update a Fantacalcio ↔ FotMob mapping.

    All fields are optional except those identifying the row; only
    non-``None`` fields will be updated.
    """

    player_fotmob_id: int | None = None
    """FotMob player ID to assign. Set to ``-1`` to clear/keep unmatched."""
    name_fotmob: str | None = None
    """FotMob player name (informational)."""
    team_fotmob: str | None = None
    """FotMob team name (optional override)."""
    canonical_role: str | None = None
    """Override canonical role (GK/DEF/MID/FWD)."""
    note: str | None = None
    """Free-text note about this override."""
    # MANTRA role overrides (optional)
    ruoli_mantra: list[str] | None = None
    """Override MANTRA roles (e.g. ["Dd", "E"])."""
    ruolo_primario: str | None = None
    """Override primary MANTRA role."""
    data_validated: bool | None = None
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
    avg_fvm: float | None = None


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
    team_fantacalcio: str | None = None
    canonical_role: str | None = None
    name_fotmob: str | None = None
    team_fotmob: str | None = None
    resolved_by: str | None = None
    note: str | None = None
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
    reliability_weight: float | None = None
    eligible_roles: list[str] = Field(default_factory=list)  # MANTRA only
    prediction_std: float | None = None  # ensemble std; drives risk_aversion penalty
    historical_overpay_ratio: float | None = None  # Picco/listino from pilastro4
    season_value: float | None = None
    start_probability: float | None = None


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
    mantra_role_quotas: dict[str, int] | None = None
    # When set, the squad is guaranteed to be able to field this module.
    # All formations in `formations` are still evaluated post-hoc and reported
    # in formation_feasibility, but only this one is a hard solver constraint.
    preferred_formation: FormationSchema | None = None
    risk_aversion: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Coefficient applied to prediction_std as a penalty on projected_score "
            "(score_eff = projected_score - risk_aversion * prediction_std). "
            "0.0 (default) = risk-neutral / legacy behaviour, no uncertainty penalty. "
            "Suggested presets: 'conservative' ~0.5-1.0 (favours low-variance players), "
            "'aggressive' = 0.0 (ranks purely on point estimate). Independent of the "
            "monte_carlo block — see MonteCarloConfigSchema for guidance on combining the two."
        ),
    )
    var_blend: float = Field(default=0.0, ge=0.0, le=1.0)
    esv_weight: float = Field(default=0.0, ge=0.0)
    hybrid_blend: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 (default) = disabled, legacy behaviour. >0 blends the MANTRA-ibrido "
            "fpIbrido signal (ml/mantra_ibrido) into the CLASSIC objective, same shape "
            "as var_blend. Requires a mantra_ibrido_results_*.json artifact to be "
            "available (local or R2); players without a match keep their base score."
        ),
    )
    # Pool pre-filter: esclude i giocatori con start_probability inferiore
    # alla soglia PRIMA della costruzione della pool passata al solver ILP.
    # `None` (default) ⇒ nessun filtro (compatibilità con richieste legacy).
    # Quando `var_blend > 0` oppure `esv_weight > 0`, lo stesso valore viene
    # propagato al VarEngine per garantire coerenza tra ranking VAR e pool.
    min_start_probability: float | None = Field(
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
        description=(
            "PER_MATCH_RATING (default, legacy) = objective uses per-match projected_score. "
            "SEASON_VALUE = objective uses season_value (expected_minutes-weighted total), "
            "which requires an ML-populated pool (season_value is null for players without "
            "an ML prediction match and falls back to PER_MATCH_RATING ranking for those rows)."
        ),
    )
    strategy_names: list[str] | None = None
    custom_strategies: list[StrategyProfileSchema] | None = None
    pool_override: list[PlayerSchema] | None = Field(default=None, max_length=500)
    monte_carlo: MonteCarloConfigSchema | None = Field(default=None)
    diversify_strategies: bool = Field(
        default=False,
        description="If true on /multi, re-solve secondary strategies excluding core of the primary when overlap is high.",
    )
    near_optimal: NearOptimalConfigSchema | None = Field(default=None)


class MonteCarloConfigSchema(_CamelModel):
    """Monte Carlo robustness block. Default off — omit for legacy deterministic ILP.

    Trade-offs:
    * ``mean_std``: 1× ILP latency, risk-adjusted point estimate (mean − λ·std).
      Always allowed on the sync path (``/optimize/multi``, ``/optimize/single``).
    * ``saa_frequency``: N× ILP latency; returns selection frequency + stability_index.
      Sync path capped by ``API_OPTIMIZER_ASYNC_THRESHOLD`` (default 50).
      Higher N must use ``POST /optimize/jobs`` (up to ``API_OPTIMIZER_MAX_SIMULATIONS``).
    * Do not combine high ``risk_aversion`` with MC without reading both effects
      (risk_aversion shrinks projected_score once; MC re-samples scores).
    """

    enabled: bool = Field(
        default=False, description="Master switch; false keeps deterministic path."
    )
    n_simulations: int = Field(
        default=200,
        ge=1,
        le=1000,
        description=(
            "SAA scenarios. Sync saa_frequency capped by API_OPTIMIZER_ASYNC_THRESHOLD; "
            "hard ceiling API_OPTIMIZER_MAX_SIMULATIONS on all paths."
        ),
    )
    mode: str = Field(
        default="saa_frequency",
        description="mean_std (fast risk-adjusted single solve) | saa_frequency (distributional).",
    )
    risk_lambda: float = Field(
        default=0.5, ge=0.0, description="Only for mean_std: mean − λ·std."
    )
    min_selection_frequency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Warn when representative players fall below this SAA frequency.",
    )
    random_seed: int = Field(
        default=42, description="Reproducible residual draws / scenario order."
    )
    timeout_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Soft SAA wall budget; 0 → API_OPTIMIZER_SAA_TIMEOUT_SECONDS default.",
    )


class SensitivityPointSchema(_CamelModel):
    value: float
    status: str
    total_score: float
    score_delta: float
    score_delta_pct: float
    jaccard_vs_baseline: float
    players_changed: int


class ParameterSensitivitySchema(_CamelModel):
    parameter: str
    points: list[SensitivityPointSchema] = Field(default_factory=list)


class SensitivityResponseSchema(_CamelModel):
    baseline_status: str
    baseline_total_score: float
    baseline_squad_size: int
    parameters: list[ParameterSensitivitySchema] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ParetoPointSchema(_CamelModel):
    risk_lambda: float
    status: str
    score: float
    risk: float
    win_probability: float | None = None
    squad_size: int
    dominated: bool = False


class ParetoResponseSchema(_CamelModel):
    points: list[ParetoPointSchema] = Field(default_factory=list)
    frontier_risk_lambdas: list[float] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    residual_source: str | None = None
    residual_using: str | None = None
    residual_rows: int | None = None
    residual_merged_rows: int | None = None


class NearOptimalConfigSchema(_CamelModel):
    enabled: bool = False
    n_alternatives: int = Field(default=3, ge=1, le=10)
    exclude_top_m: int = Field(default=2, ge=1, le=10)
    max_score_drop_pct: float = Field(default=0.15, ge=0.0, le=1.0)


class NearOptimalAlternativeSchema(_CamelModel):
    excluded_player_ids: list[str]
    score_delta: float
    score_delta_pct: float
    squad: list[SquadPlayerSchema]
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
    error: str | None = None
    result: OptimizationResultSchema | None = None
    monte_carlo_summary: MonteCarloSummarySchema | None = None


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
    win_probability: float | None = None
    monte_carlo_summary: MonteCarloSummarySchema | None = None
    near_optimal: list[NearOptimalAlternativeSchema] = Field(default_factory=list)


class MultiStrategyResultSchema(_CamelModel):
    """Output di ``POST /optimize/multi``: una entry per strategia richiesta."""

    results: dict[str, OptimizationResultSchema]
    monte_carlo_summary: MonteCarloSummarySchema | None = None
    diversity: DiversityMetricsSchema | None = None


class StrategyProfileSchema(_CamelModel):
    """Profilo di strategia di ottimizzazione (read-only, informativo)."""

    name: str
    role_weight: dict[str, float]
    min_budget_share_by_roles: tuple[list[str], float] | None = None
    max_top_tier_players: int | None = None
    top_tier_cost_threshold: float | None = None


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
    ruleset: str = "CLASSIC"
    """Ruleset dell'asta: ``"CLASSIC"`` (4 ruoli, default) o ``"MANTRA"``
    (12 ruoli multi-slot). Stesso pattern di :class:`OptimizationRequest`.
    Quando ``ruleset="MANTRA"`` e ``role_quotas`` resta al default CLASSIC,
    il dominio applica automaticamente ``MANTRA_DEFAULT_QUOTAS``.
    """
    market_drift_config: MarketDriftConfigSchema = MarketDriftConfigSchema()
    alternatives_config: AlternativesConfigSchema = AlternativesConfigSchema()
    use_inflation_baseline: bool = False
    inflation_config: InflationConfigSchema | None = None
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

    hybrid_blend: float = 0.0
    """WS3 #2: weight in [0, 1] of fpIbrido signal in VarEngine. 0 = off."""

    @field_validator("reference_budget", "budget_initial")
    @classmethod
    def _validate_positive_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v

    @field_validator("ruleset")
    @classmethod
    def _validate_ruleset(cls, v: str) -> str:
        if v not in ("CLASSIC", "MANTRA"):
            raise ValueError(f"ruleset must be CLASSIC or MANTRA, got {v!r}")
        return v


class AuctionPlayerSchema(_CamelModel):
    """Player passato al backend per inizializzare il pool d'asta."""

    player_id: str
    name: str
    role: str  # P | D | C | A (classic); MANTRA uses eligible_roles
    real_team: str
    cost: int
    projected_score: float
    season_value: float | None = None
    start_probability: float | None = None
    eligible_roles: list[str] | None = None
    """MANTRA only: list of Mantra role codes this player can fill
    (e.g. ``["Dd", "E"]``). Ignored under CLASSIC."""


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
    player_pool: list[AuctionPlayerSchema] | None = None


class InitializeAuctionResponse(_CamelModel):
    """Response di ``POST /auction/init``: id di sessione generato lato server."""

    session_id: str


class RecordAssignmentRequest(_CamelModel):
    """Request body per ``POST /auction/{session_id}/record``."""

    player_id: str
    winner_participant_id: str
    final_price: int
    assigned_slot: str | None = None
    """MANTRA only: explicit role slot filled by this assignment
    (e.g. ``"Dd"``). If omitted the orchestrator auto-picks among the
    player's eligible roles with residual quota. Ignored under CLASSIC."""


class RecordAssignmentResponse(_CamelModel):
    """Response di ``POST /auction/{session_id}/record``.

    Su success, include uno snapshot minimo (sequence_number,
    price_index_after); su rifiuto, popolati ``rejection_code`` e
    ``rejection_reason``.
    """

    success: bool
    sequence_number: int | None = None
    price_index_after: float | None = None
    rejection_code: str | None = None
    rejection_reason: str | None = None


class ProjectionResponse(_CamelModel):
    """Response di ``GET /auction/{session_id}/projection/{player_id}``."""

    player_id: str
    expected_price: float
    tier: str  # LOW | MID | TOP


class AlternativesRequest(_CamelModel):
    """Request body opzionale per ``GET /auction/{session_id}/alternatives``."""

    config: AlternativesConfigSchema | None = None


class AuctionPlayerSummarySchema(_CamelModel):
    """Player serializzato nelle risposte di summary."""

    player_id: str
    name: str
    real_team: str
    role: str
    cost: int
    projected_score: float
    season_value: float | None = None
    start_probability: float | None = None
    eligible_roles: list[str] | None = None


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
    assigned_slot: str | None = None
    """Slot MANTRA effettivamente occupato (coincide con ``role`` in CLASSIC)."""


class AuctionSummarySchema(_CamelModel):
    """Response di ``GET /auction/{session_id}/summary``."""

    participants: list[AuctionParticipantStateSchema]
    assignments: list[AssignmentRecordSchema]
    price_index: dict[str, dict[str, float]]
    completion_probability: dict[str, float] | None = None
    """WS3 #1: participant_id → P(complete roster | residual budget)."""


class AlternativesResponse(_CamelModel):
    """Response di ``GET /auction/{session_id}/alternatives/{player_id}``."""

    target_player_id: str
    low_cost_alternative: AuctionPlayerSummarySchema | None = None
    closest_alternative: AuctionPlayerSummarySchema | None = None
    reason_if_none: str | None = None
    diversified_alternatives: list[AuctionPlayerSummarySchema] = []
    """WS3 #3: mini-Pareto diversified candidates."""
    max_affordable_bid: int | None = None
    """WS3 #4: credit-reserve max bid for the requested participant."""
    strategy_price_cap: int | None = None
    """WS3 #5: strategy-weighted price threshold."""


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
    season_value: float | None = None
    start_probability: float | None = None


class VarRankingResponse(_CamelModel):
    """Response of ``GET /auction/{session_id}/var-ranking``."""

    session_id: str
    items: list[VarRankingItemSchema]
    using_live_prices: bool


# ---------------------------------------------------------------------------
# Monte Carlo auction simulation (stateless)
# ---------------------------------------------------------------------------


class BidderPolicySchema(_CamelModel):
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    inflation_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    max_overpay_ratio: float = Field(default=1.2, ge=1.0)
    min_residual_credits_per_slot: float = Field(default=1.5, ge=0.0)
    all_in_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    budget_elasticity: float = Field(default=0.4, ge=0.0, le=1.0)
    var_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    team_strength_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    prefer_alternatives: bool = True
    prefer_low_cost_alternative: bool = False
    rebid_trigger_pct_above_expected: float = Field(default=0.12, ge=0.0)
    budget_share_by_role: dict[str, float] | None = None
    phase_bias: str | None = None
    prefer_young_players: bool = False
    max_age_preference: int | None = None
    prefer_high_start_probability: bool = False
    min_start_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    prefer_high_variance: bool = False
    prefer_multi_role: bool = False
    min_num_roles: int | None = Field(default=None, ge=1)
    budget_share_by_block: dict[str, float] | None = None
    max_top_tier_count: int | None = Field(default=None, ge=0)
    target_top_tier_count: int | None = Field(default=None, ge=0)
    avoid_top_tier_early: bool = False
    adaptive: bool = False
    adapt_on: list[str] | None = None


class BidderProfileSchema(_CamelModel):
    participant_id: str
    policy: BidderPolicySchema = Field(default_factory=BidderPolicySchema)


class AuctionSimulationConfigSchema(_CamelModel):
    n_simulations: int = Field(default=200, ge=1, le=500)
    random_seed: int = Field(default=42)
    price_noise_std_ratio: float = Field(default=0.15, ge=0.0)
    timeout_seconds: float = Field(default=0.0, ge=0.0)
    min_bid_step: int = Field(default=1, ge=1)


class SimulateAuctionRequest(_CamelModel):
    season_start: int
    participants: list[AuctionParticipantSetupSchema]
    config: AuctionConfigSchema
    player_pool: list[AuctionPlayerSchema] | None = None
    bidder_profiles: list[BidderProfileSchema] = Field(default_factory=list)
    sim_config: AuctionSimulationConfigSchema = Field(
        default_factory=AuctionSimulationConfigSchema
    )


class ParticipantSimStatsSchema(_CamelModel):
    spend_p10: float
    spend_p50: float
    spend_p90: float
    esv_total_p10: float
    esv_total_p50: float
    esv_total_p90: float
    completion_probability: float
    squad_composition_mode: dict[str, int] = Field(default_factory=dict)


class PlayerAcquisitionStatsSchema(_CamelModel):
    prob: float
    avg_price: float


class AuctionSimulationResponse(_CamelModel):
    n_completed: int
    per_participant: dict[str, ParticipantSimStatsSchema]
    price_index_drift_p50: dict[str, dict[str, float]] = Field(default_factory=dict)
    player_acquisition_probability: dict[str, PlayerAcquisitionStatsSchema] = Field(
        default_factory=dict
    )
    wall_time_seconds: float
    warnings: list[str] = Field(default_factory=list)
