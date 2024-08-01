ALTER TABLE datum ADD COLUMN textblob varchar;
ALTER TABLE annotation ADD COLUMN textblob varchar;
ALTER TABLE annotation ADD COLUMN context jsonb;