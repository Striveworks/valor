ALTER TABLE evaluation ADD COLUMN dataset_names jsonb;

ALTER TABLE evaluation
DROP CONSTRAINT evaluation_model_name_datum_filter_parameters_key;

UPDATE evaluation
SET dataset_names = datum_filter->'dataset_names';

UPDATE evaluation
SET datum_filter = jsonb_set(datum_filter, '{dataset_names}', 'null'::jsonb);

ALTER TABLE evaluation ADD CONSTRAINT evaluation_unique_constraint
UNIQUE (model_name, datum_filter, parameters, dataset_names);