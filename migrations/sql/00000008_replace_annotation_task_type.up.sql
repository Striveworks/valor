ALTER TABLE annotation DROP COLUMN task_type;
ALTER TABLE annotation ADD COLUMN is_instance_segmentation boolean;
ALTER TABLE annotation ADD COLUMN implied_task_types jsonb;