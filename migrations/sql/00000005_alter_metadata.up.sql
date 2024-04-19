UPDATE dataset SET meta = subquery1.value
FROM (
    SELECT
        id,
        jsonb_object_agg(subquery2.key, subquery2.value) AS value
    FROM (
        SELECT
            subquery3.key AS key,
            CASE
                WHEN jsonb_typeof(subquery3.value)::text = 'object'
                THEN jsonb_build_object('type', element.key, 'value', element.value)
                ELSE subquery3.value
            END AS value,
            subquery3.id
        FROM (
            SELECT key, value, id
            FROM dataset,
            LATERAL jsonb_each(meta)
        ) AS subquery3
        LEFT JOIN LATERAL (
            SELECT key, value FROM jsonb_each(
                CASE
                    WHEN jsonb_typeof(subquery3.value) = 'object'
                    THEN subquery3.value
                END
            )
        ) AS element ON true
    ) AS subquery2
    GROUP BY id
) AS subquery1
WHERE dataset.id = subquery1.id;

UPDATE model SET meta = subquery1.value
FROM (
    SELECT
        id,
        jsonb_object_agg(subquery2.key, subquery2.value) AS value
    FROM (
        SELECT
            subquery3.key AS key,
            CASE
                WHEN jsonb_typeof(subquery3.value)::text = 'object'
                THEN jsonb_build_object('type', element.key, 'value', element.value)
                ELSE subquery3.value
            END AS value,
            subquery3.id
        FROM (
            SELECT key, value, id
            FROM model,
            LATERAL jsonb_each(meta)
        ) AS subquery3
        LEFT JOIN LATERAL (
            SELECT key, value FROM jsonb_each(
                CASE
                    WHEN jsonb_typeof(subquery3.value) = 'object'
                    THEN subquery3.value
                END
            )
        ) AS element ON true
    ) AS subquery2
    GROUP BY id
) AS subquery1
WHERE model.id = subquery1.id;

UPDATE datum SET meta = subquery1.value
FROM (
    SELECT
        id,
        jsonb_object_agg(subquery2.key, subquery2.value) AS value
    FROM (
        SELECT
            subquery3.key AS key,
            CASE
                WHEN jsonb_typeof(subquery3.value)::text = 'object'
                THEN jsonb_build_object('type', element.key, 'value', element.value)
                ELSE subquery3.value
            END AS value,
            subquery3.id
        FROM (
            SELECT key, value, id
            FROM datum,
            LATERAL jsonb_each(meta)
        ) AS subquery3
        LEFT JOIN LATERAL (
            SELECT key, value FROM jsonb_each(
                CASE
                    WHEN jsonb_typeof(subquery3.value) = 'object'
                    THEN subquery3.value
                END
            )
        ) AS element ON true
    ) AS subquery2
    GROUP BY id
) AS subquery1
WHERE datum.id = subquery1.id;

UPDATE annotation SET meta = subquery1.value
FROM (
    SELECT
        id,
        jsonb_object_agg(subquery2.key, subquery2.value) AS value
    FROM (
        SELECT
            subquery3.key AS key,
            CASE
                WHEN jsonb_typeof(subquery3.value)::text = 'object'
                THEN jsonb_build_object('type', element.key, 'value', element.value)
                ELSE subquery3.value
            END AS value,
            subquery3.id
        FROM (
            SELECT key, value, id
            FROM annotation,
            LATERAL jsonb_each(meta)
        ) AS subquery3
        LEFT JOIN LATERAL (
            SELECT key, value FROM jsonb_each(
                CASE
                    WHEN jsonb_typeof(subquery3.value) = 'object'
                    THEN subquery3.value
                END
            )
        ) AS element ON true
    ) AS subquery2
    GROUP BY id
) AS subquery1
WHERE annotation.id = subquery1.id;
