ALTER TABLE evaluation ADD COLUMN dataset_names jsonb;

UPDATE evaluation
SET dataset_names = datum_filter->'dataset_names';
