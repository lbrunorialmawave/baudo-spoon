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
  /** Gruppo Esperti breakdown (source: "gruppo_esperti" only) — null for other sources. */
  titolarita: number | null;
  media_voto: number | null;
  salute: number | null;
  /** Label of the 4th stat: "Bonus" / "No Gol" (keepers) / "Porta inviolata". */
  bonus_label: string | null;
  bonus_value: number | null;
  /** Overall total, out of 50. */
  totale: number | null;
  /** Same value as `rating`, uncompressed (1-10 instead of 1-5 stars). */
  consiglio_esperti_raw: number | null;
  birth_year: number | null;
  /** Set when `comment` is just a pointer (e.g. "Vedi possibili sorprese") — the referenced section. */
  cross_reference_section: string | null;
  cross_reference_text: string | null;
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
