ALTER TABLE annotation ADD COLUMN is_instance boolean;
ALTER TABLE annotation ADD COLUMN implied_task_types jsonb;

UPDATE annotation
SET implied_task_types = jsonb_build_array(task_type);

ALTER TABLE annotation DROP COLUMN task_type;