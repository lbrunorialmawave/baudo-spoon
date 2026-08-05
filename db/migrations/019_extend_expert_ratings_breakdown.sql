-- Migration: extend expert_ratings with the full Gruppo Esperti breakdown.
--
-- forum.gruppoesperti.it player entries carry a full stat line (Titolarità,
-- Media voto, Salute, Bonus, Consiglio Esperti, TOTALE) plus a possible
-- cross-reference to another section of the post (e.g. a comment that's
-- just "Vedi possibili sorprese"). Previously only Consiglio Esperti
-- (compressed to a 1-5 `rating`) and the raw comment were kept; everything
-- else was parsed and discarded. All columns are nullable since only the
-- "gruppo_esperti" source populates them.

ALTER TABLE expert_ratings
    ADD COLUMN IF NOT EXISTS titolarita               SMALLINT,
    ADD COLUMN IF NOT EXISTS media_voto                SMALLINT,
    ADD COLUMN IF NOT EXISTS salute                    SMALLINT,
    ADD COLUMN IF NOT EXISTS bonus_label                VARCHAR(50),
    ADD COLUMN IF NOT EXISTS bonus_value                SMALLINT,
    ADD COLUMN IF NOT EXISTS totale                     SMALLINT,
    ADD COLUMN IF NOT EXISTS consiglio_esperti_raw       SMALLINT,
    ADD COLUMN IF NOT EXISTS birth_year                  SMALLINT,
    ADD COLUMN IF NOT EXISTS cross_reference_section     VARCHAR(50),
    ADD COLUMN IF NOT EXISTS cross_reference_text        TEXT;

COMMENT ON COLUMN expert_ratings.titolarita IS
    'Gruppo Esperti "Titolarità" score, 1-10.';
COMMENT ON COLUMN expert_ratings.media_voto IS
    'Gruppo Esperti "Media voto" score, 1-10.';
COMMENT ON COLUMN expert_ratings.salute IS
    'Gruppo Esperti "Salute" (fitness/injury-free) score, 1-10.';
COMMENT ON COLUMN expert_ratings.bonus_label IS
    'Label of the 4th stat, which varies by role/curator: "Bonus", "No Gol" (keepers), "Porta inviolata".';
COMMENT ON COLUMN expert_ratings.bonus_value IS
    'Value of the bonus_label stat, 1-10.';
COMMENT ON COLUMN expert_ratings.totale IS
    'Gruppo Esperti overall total, out of 50.';
COMMENT ON COLUMN expert_ratings.consiglio_esperti_raw IS
    'Raw "Consiglio Esperti" score, 1-10 (rating column is the same value compressed to 1-5 stars).';
COMMENT ON COLUMN expert_ratings.birth_year IS
    'Player birth year, as printed next to the name in the source post.';
COMMENT ON COLUMN expert_ratings.cross_reference_section IS
    'Set when comment is just a pointer (e.g. "Vedi possibili sorprese") — the referenced section marker (e.g. POSSIBILI.SORPRESE).';
COMMENT ON COLUMN expert_ratings.cross_reference_text IS
    'Text of the section referenced by cross_reference_section, resolved at scrape time.';
