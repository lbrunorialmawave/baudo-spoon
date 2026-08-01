// NOTE: expert_ratings rows come back as raw snake_case (the router builds
// them with `dict(r._mapping)` from a SQL text query, not a _CamelModel).

export interface ExpertRating {
  id: number;
  player_id: string;
  source: string;
  expert_name: string | null;
  rating: number | null;
  comment: string | null;
  matchday: number | null;
  season_start: number;
  url: string | null;
  scraped_at: string;
}

export interface PlayerExpertRatingsResponse {
  player_fotmob_id: number;
  total_ratings: number;
  average_rating: number | null;
  ratings: ExpertRating[];
}
