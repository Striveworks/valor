ALTER TABLE datum ADD COLUMN text text;
ALTER TABLE annotation ADD COLUMN text text;
ALTER TABLE annotation ADD COLUMN context jsonb;
ALTER TABLE groundtruth ALTER COLUMN label_id DROP NOT NULL;
ALTER TABLE prediction ALTER COLUMN label_id DROP NOT NULL;