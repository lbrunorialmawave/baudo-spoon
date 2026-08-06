-- Migration: drop team_season_odds.
--
-- Removes the table backing the Snai/Odds API "Vincitore Serie A" scrapers
-- (scraper/snai_odds.py, scraper/odds_api.py), which have been removed —
-- Snai blocks TLS from datacenter IPs and The Odds API was a paid stopgap
-- replacement. PS_corretto's snai_winner_odds component (ml/mantra) was
-- never fed real data by the runner, so nothing downstream reads this table.

DROP TABLE IF EXISTS team_season_odds;
