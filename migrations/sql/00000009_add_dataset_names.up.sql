ALTER TABLE evaluation ADD COLUMN dataset_names jsonb;

UPDATE evaluation
SET dataset_names = datum_filter->'dataset_names';

UPDATE evaluation
SET datum_filter = jsonb_set(datum_filter, '{dataset_names}', 'null'::jsonb);
