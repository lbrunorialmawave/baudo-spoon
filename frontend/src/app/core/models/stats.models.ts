// NOTE: All fields are snake_case.
// The stats/matches/leagues/seasons routers do NOT use alias_generator=to_camel.

export interface League {
  id: number;
  name: string;
  comp_id: string;
  slug: string;
}

export interface Season {
  id: number;
  season_start: number;
  season_label: string;
  scraped_at: string | null;
  league: League;
}

export interface PlayerSeasonStat {
  id: number;
  fotmob_season_id: number;
  stat_category: string;
  rank: number | null;
  player_fotmob_id: number;
  player_name: string;
  team_fotmob_id: number | null;
  team_name: string | null;
  value: string | null;
  ingested_at: string;
  season: Season;
}

export interface TeamSeasonStat {
  id: number;
  fotmob_season_id: number;
  stat_category: string;
  rank: number | null;
  team_fotmob_id: number;
  team_name: string;
  value: string | null;
  ingested_at: string;
  season: Season;
}

export interface MatchStat {
  id: number;
  match_date: string | null;
  round_num: number | null;
  match_name: string;
  score: string | null;
  status: string | null;
  url: string | null;
  team: string;
  side: string | null;
  opponent: string | null;
  goals_scored: number | null;
  goals_conceded: number | null;
  points: number | null;
  stats: Record<string, unknown>;
  ingested_at: string;
  season: Season;
}
