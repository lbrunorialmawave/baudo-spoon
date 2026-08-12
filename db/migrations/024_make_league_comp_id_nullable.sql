-- Allow dynamic (uncatalogued) leagues from careerHistory without known comp_id.
ALTER TABLE leagues
    ALTER COLUMN comp_id DROP NOT NULL;
