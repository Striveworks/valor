UPDATE evaluation
SET datum_filter->'dataset_names' = dataset_names;

ALTER TABLE evaluation DROP COLUMN dataset_names;