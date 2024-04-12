ALTER TABLE multipolygon ADD multipolygon geometry(MultiPolygon);
UPDATE evaluation SET datum_filter = jsonb_set(datum_filter, '{multipolygon_area}', 'null', true);
UPDATE evaluation SET datum_filter = jsonb_set(datum_filter, '{require_multipolygon}', 'null', true);
