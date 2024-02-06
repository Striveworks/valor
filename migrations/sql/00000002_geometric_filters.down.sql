UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{annotation_types}',
    (
        SELECT jsonb_agg(DISTINCT value)
        FROM evaluation, jsonb_array_elements(
            jsonb_build_array(
                CASE
                    WHEN datum_filter->'require_bounding_box' = 'true' THEN 'box'
                END,
                CASE
                    WHEN datum_filter->'require_polygon' = 'true' THEN 'polygon'
                END,
                CASE
                    WHEN datum_filter->'require_multipolygon' = 'true' THEN 'multipolygon'
                END,
                CASE
                    WHEN datum_filter->'require_raster' = 'true' THEN 'raster'
                END
            )
        ) as elements(value)
        WHERE value != 'null'
    )
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{annotation_geometric_area}',
    (
        select jsonb_agg(DISTINCT value)
        from evaluation, jsonb_array_elements(
            (datum_filter -> 'bounding_box_area')
            || (datum_filter -> 'polygon_area')
            || (datum_filter -> 'multipolygon_area' )
            || (datum_filter -> 'raster_area' )
        ) as elements(value)
        where value != 'null'
    )
);

UPDATE evaluation
SET datum_filter = datum_filter - 'require_bounding_box';

UPDATE evaluation
SET datum_filter = datum_filter - 'require_polygon';

UPDATE evaluation
SET datum_filter = datum_filter - 'require_multipolygon';

UPDATE evaluation
SET datum_filter = datum_filter - 'require_raster';

UPDATE evaluation
SET datum_filter = datum_filter - 'bounding_box_area';

UPDATE evaluation
SET datum_filter = datum_filter - 'polygon_area';

UPDATE evaluation
SET datum_filter = datum_filter - 'multipolygon_area';

UPDATE evaluation
SET datum_filter = datum_filter - 'raster_area';
