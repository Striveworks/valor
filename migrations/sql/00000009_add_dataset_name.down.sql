ALTER TABLE evaluation
DROP CONSTRAINT evaluation_unique_constraint;

UPDATE evaluation
SET filters = jsonb_set(filters, '{dataset_names}', dataset_names::jsonb);

ALTER TABLE evaluation DROP COLUMN dataset_names;

ALTER TABLE evaluation
RENAME COLUMN filters TO datum_filter;

ALTER TABLE evaluation ADD CONSTRAINT evaluation_model_name_datum_filter_parameters_key
UNIQUE (model_name, datum_filter, parameters);
