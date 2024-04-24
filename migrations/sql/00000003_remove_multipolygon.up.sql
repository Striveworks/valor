ALTER TABLE annotation DROP COLUMN multipolygon;
UPDATE evaluation SET datum_filter = datum_filter - 'multipolygon_area';
UPDATE evaluation SET datum_filter = datum_filter - 'require_multipolygon';
