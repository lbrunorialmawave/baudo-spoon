from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict
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
