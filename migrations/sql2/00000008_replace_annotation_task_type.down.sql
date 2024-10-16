ALTER TABLE annotation ADD COLUMN task_type varchar;

UPDATE annotation
SET task_type = implied_task_types->>0;

ALTER TABLE annotation DROP COLUMN is_instance;
ALTER TABLE annotation DROP COLUMN implied_task_types;
