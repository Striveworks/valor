ALTER TABLE datum ADD COLUMN text text;
ALTER TABLE annotation ADD COLUMN text text;
ALTER TABLE annotation ADD COLUMN context jsonb;