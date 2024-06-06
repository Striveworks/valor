ALTER TABLE annotation ADD COLUMN task_type varchar;
ALTER TABLE annotation DROP COLUMN is_instance_segmentation;
ALTER TABLE annotation DROP COLUMN implied_task_types;
