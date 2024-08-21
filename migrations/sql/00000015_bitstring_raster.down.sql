ALTER TABLE annotation ADD COLUMN raster raster;

-- TODO add bitmask to raster conversion

ALTER TABLE annotation DROP COLUMN bitmask_id;

drop index ix_bitmask_id;
drop TABLE bitmask;
