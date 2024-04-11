-- note: if you've already created a PrecisionRecallCurve metric in your db, the line below will fail with ERROR:  cannot cast jsonb object to type double precision
-- you'll have to delete all metrics with type = "PrecisionRecallCurve" before running this line
ALTER TABLE if exists metric ALTER COLUMN value TYPE double precision USING value::double precision