ALTER TABLE evaluation
DROP CONSTRAINT evaluation_model_name_datum_filter_parameters_key;

ALTER TABLE evaluation
RENAME COLUMN datum_filter TO filters;

ALTER TABLE evaluation ADD COLUMN dataset_names jsonb;

UPDATE evaluation
SET dataset_names = filters->'dataset_names';

UPDATE evaluation
SET filters = jsonb_set(filters, '{dataset_names}', 'null'::jsonb);

ALTER TABLE evaluation ADD CONSTRAINT evaluation_unique_constraint
UNIQUE (model_name, filters, parameters, dataset_names);