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

/** Row shape from GET /experts/ratings/for-season/{season} — same as
 * ExpertRating plus fantacalcio_id, pulled server-side from the
 * `fc-{id}` player_id convention so it can key a client-side map the same
 * way mantraMap / matchdayStatusMap are keyed. */
export interface ExpertRatingWithFantacalcioId extends ExpertRating {
  fantacalcio_id: number;
}

export interface SeasonExpertRatingsResponse {
  season_start: number;
  total: number;
  items: ExpertRatingWithFantacalcioId[];
}
