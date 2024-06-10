UPDATE evaluation
SET datum_filter = jsonb_set(datum_filter, '{dataset_names}', dataset_names::jsonb);

ALTER TABLE evaluation DROP COLUMN dataset_names;
