-- Migration: scope expert_ratings' uniqueness by season_start too.
--
-- uq_expert_rating was UNIQUE (player_id, source, expert_name, matchday) —
-- missing season_start. player_id is "fc-{fantacalcio_id}", and
-- fantacalcio_id is a season-scoped Fantacalcio listone id that gets
-- reassigned to a *different* physical player from one season to the next.
-- When that happens, a new season's upsert collides with the previous
-- season's row on this constraint; the scraper's ON CONFLICT DO UPDATE
-- (scraper/gruppo_esperti.py) doesn't touch season_start, so the row's
-- content silently updates to the new player/season while season_start
-- stays stuck at whatever it was first inserted with — the row becomes
-- unreachable from GET /experts/ratings/for-season/{current_season} even
-- though it was "successfully" persisted moments ago.
--
-- Fixing the constraint alone is not enough to repair rows already
-- corrupted this way (their season_start is already wrong and nothing here
-- can recover which season it should have been) — but it makes the bug
-- stop recurring: a same-player re-scrape within a season still upserts in
-- place, while a genuinely different season now inserts its own row
-- instead of overwriting the wrong one.

ALTER TABLE expert_ratings
    DROP CONSTRAINT IF EXISTS uq_expert_rating;

ALTER TABLE expert_ratings
    ADD CONSTRAINT uq_expert_rating
        UNIQUE (player_id, source, expert_name, matchday, season_start);
