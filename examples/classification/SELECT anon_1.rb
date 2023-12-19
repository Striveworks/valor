SELECT anon_1.pred_label_value,
    label.value,
    count(*) AS count_1

select *  
FROM (
        SELECT groundtruth.id AS id,
            groundtruth.annotation_id AS annotation_id,
            groundtruth.label_id AS label_id,
            annotation.datum_id AS datum_id
        FROM dataset
            JOIN datum ON datum.dataset_id = dataset.id
            JOIN annotation ON annotation.datum_id = datum.id
            JOIN groundtruth ON groundtruth.annotation_id = annotation.id
            JOIN label ON label.id = groundtruth.label_id
        WHERE dataset.name = 'breast-cancer-train'
            AND annotation.task_type = 'classification'
            AND label.key = 'class'
    ) AS anon_2
    JOIN (
        SELECT label.value AS pred_label_value,
            datum.id AS datum_id
        FROM (
            SELECT min(anon_4.id) AS min_id
            FROM (
                    SELECT prediction.id AS id,
                        prediction.annotation_id AS annotation_id,
                        prediction.label_id AS label_id,
                        prediction.score AS score
                    FROM model
                        JOIN annotation ON annotation.model_id = model.id
                        JOIN prediction ON prediction.annotation_id = annotation.id
                        JOIN label ON label.id = prediction.label_id
                        JOIN datum ON datum.id = annotation.datum_id
                        JOIN dataset ON dataset.id = datum.dataset_id
                    WHERE model.name = 'breast-cancer-linear-model'
                        AND dataset.name = 'breast-cancer-train'
                        AND label.key = 'class'
                        AND annotation.task_type = 'classification'
                ) AS anon_4
                JOIN annotation ON annotation.id = anon_4.annotation_id
                JOIN datum ON annotation.datum_id = datum.id
                JOIN (
                    SELECT max(prediction.score) AS max_score,
                        datum.id AS datum_id
                    FROM model
                        JOIN annotation ON annotation.model_id = model.id
                        JOIN prediction ON prediction.annotation_id = annotation.id
                        JOIN label ON label.id = prediction.label_id
                        JOIN datum ON datum.id = annotation.datum_id
                        JOIN dataset ON dataset.id = datum.dataset_id
                    WHERE model.name = 'breast-cancer-linear-model'
                        AND dataset.name = 'breast-cancer-train'
                        AND label.key = 'class'
                        AND annotation.task_type = 'classification'
                    GROUP BY datum.id
                ) AS anon_5 ON datum.id = anon_5.datum_id
                AND anon_4.score = anon_5.max_score
                JOIN label ON label.id = anon_4.label_id
            GROUP BY datum.id
        ) AS anon_3
        JOIN prediction ON prediction.id = anon_3.min_id
        JOIN annotation ON annotation.id = prediction.annotation_id
        JOIN datum ON datum.id = annotation.datum_id
        JOIN (
            SELECT max(prediction.score) AS max_score,
                datum.id AS datum_id
            FROM model
                JOIN annotation ON annotation.model_id = model.id
                JOIN prediction ON prediction.annotation_id = annotation.id
                JOIN label ON label.id = prediction.label_id
                JOIN datum ON datum.id = annotation.datum_id
                JOIN dataset ON dataset.id = datum.dataset_id
            WHERE model.name = 'breast-cancer-linear-model'
                AND dataset.name = 'breast-cancer-train'
                AND label.key = 'class'
                AND annotation.task_type = 'classification'
            GROUP BY datum.id
        ) AS anon_5 ON prediction.score = anon_5.max_score
        AND datum.id = anon_5.datum_id
        JOIN label ON label.id = prediction.label_id
    ) AS anon_1 ON anon_1.datum_id = anon_2.datum_id
    JOIN label ON label.id = anon_2.label_id
WHERE label.key = 'class'
GROUP BY anon_1.pred_label_value,
    label.value