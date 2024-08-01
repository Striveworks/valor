ALTER TABLE datum ADD COLUMN text varchar;
ALTER TABLE annotation ADD COLUMN text varchar;
ALTER TABLE annotation ADD COLUMN context jsonb;