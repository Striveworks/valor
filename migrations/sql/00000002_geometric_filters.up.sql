UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{require_bounding_box}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'box'
        ) THEN 'true'::jsonb
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{require_polygon}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'polygon'
        ) THEN 'true'::jsonb
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{require_multipolygon}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'multipolygon'
        ) THEN 'true'::jsonb
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{require_raster}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'raster'
        ) THEN 'true'::jsonb
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{bounding_box_area}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'box'
        ) THEN datum_filter -> 'annotation_geometric_area'
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{polygon_area}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'polygon'
        ) THEN datum_filter -> 'annotation_geometric_area'
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{multipolygon_area}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'multipolygon'
        ) THEN datum_filter -> 'annotation_geometric_area'
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = jsonb_set(
    datum_filter,
    '{raster_area}',
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM evaluation,
                jsonb_array_elements_text(evaluation.datum_filter -> 'annotation_types') AS elements(value)
            WHERE elements.value = 'raster'
        ) THEN datum_filter -> 'annotation_geometric_area'
        ELSE 'null'::jsonb
    END
);

UPDATE evaluation
SET datum_filter = datum_filter - 'annotation_types';

UPDATE evaluation
SET datum_filter = datum_filter - 'annotation_geometric_area';
