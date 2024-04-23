-- uses jsonb rather than text[] because we want the array to be able to contain either strings or floats
ALTER TABLE if exists annotation ADD COLUMN ranking jsonb;