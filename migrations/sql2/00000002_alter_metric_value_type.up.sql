ALTER TABLE if exists metric ALTER COLUMN value TYPE JSONB USING (value)::text::jsonb;
