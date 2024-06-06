ALTER TABLE annotation ADD COLUMN is_instance boolean;
ALTER TABLE annotation ADD COLUMN implied_task_types jsonb;

UPDATE annotation
SET implied_task_types = jsonb_build_array(task_type);

UPDATE annotation
SET is_instance = CASE WHEN task_type = 'object-detection' THEN TRUE ELSE FALSE END;

ALTER TABLE annotation DROP COLUMN task_type;