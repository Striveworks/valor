UPDATE annotation SET raster = ST_SetUpperLeft(raster, 0, 0) WHERE raster IS NOT NULL;
