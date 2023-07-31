from velour_api.enums import AnnotationType


def get_dataset_id_from_datum_id(datum_id: str):
    """SQL subquery to retrieve dataset id from datum id."""
    return f"(select dataset_id from datum where id = {datum_id} limit 1)"


def filter_by_label_key(tablename, label_key):
    """SQL query to filter labels by category."""

    return f"""
    SELECT *
    FROM {tablename}
    WHERE (select key from label where id = label_id) = '{label_key}'
    """


def join_labels(
    subquery: str,
    label_table: str,
    column: str,
    label_key: str,
    is_prediction: bool = False,
):
    """SQL query to join labels to annotations."""

    return f"""
    SELECT annotation.*, label_id{', score' if is_prediction else ''}
    FROM ({subquery}) AS annotation
    JOIN ({filter_by_label_key(label_table, label_key)}) as subquery
    ON annotation.id = subquery.{column}
    """


def join_tables(
    gt_subquery: str,
    pd_subquery: str,
    datatype: AnnotationType = AnnotationType.BOX,
):
    """SQL query to join labeled table data."""

    return f"""
        WITH
        gt AS ({gt_subquery}),
        pd AS ({pd_subquery})
        SELECT
            gt.datum_id,
            gt.id as gt_id,
            pd.id as pd_id,
            gt.label_id as gt_label_id,
            pd.label_id as pd_label_id,
            gt.{datatype.value} as gt_{datatype.value},
            pd.{datatype.value} as pd_{datatype.value},
            pd.score
        FROM gt
        FULL JOIN pd
        ON gt.datum_id = pd.datum_id AND gt.label_id = pd.label_id
    """


def compute_iou(subquery: str, datatype: AnnotationType = AnnotationType.BOX):
    """SQL query to generate iou table from a joint table."""

    if datatype == AnnotationType.BOX or datatype == AnnotationType.POLYGON:
        return f"""
        SELECT
        datum_id,
        gt_id,
        pd_id,
        gt_label_id,
        pd_label_id,
        score,
        ST_Area(ST_Intersection(gt_{datatype.value}, pd_{datatype.value})) / ST_Area(ST_Union(gt_{datatype.value}, pd_{datatype.value})) iou
        FROM ({subquery}) AS subquery
        """
    elif datatype == AnnotationType.RASTER:
        return f"""
        SELECT
        datum_id,
        gt_id,
        pd_id,
        gt_label_id,
        pd_label_id,
        score,
        ST_Count(ST_Intersection(gt_{datatype.value}, pd_{datatype.value})) /
        (ST_Count(gt_{datatype.value}) + ST_Count(pd_{datatype.value}) - ST_Count(ST_Intersection(gt_{datatype.value}, pd_{datatype.value}))) iou
        FROM ({subquery}) AS subquery
        """
    else:
        raise NotImplementedError(
            f"AnnotationType {datatype} not implemented."
        )


def function_find_ranked_pairs():
    """SQL create function to find ranked ground truth/prediction pairs."""

    return """
    CREATE OR REPLACE FUNCTION find_ranked_pairs()
    RETURNS TABLE (label_id int, gt_id int, pd_id int, score float, iou float) AS
    $BODY$
        DECLARE
            row record;
        BEGIN

            CREATE TEMPORARY TABLE pairs (
                label_id int not null,
                gtid int not null,
                pdid int not null,
                score float not null,
                iou float not null,
                UNIQUE (gtid),
                UNIQUE (pdid)
            );

            FOR row IN (
                SELECT  iou.gt_label_id as label_id, iou.gt_id, iou.pd_id, iou.score, iou.iou
                FROM iou
                WHERE iou.gt_id IS NOT NULL AND iou.pd_id IS NOT NULL AND iou.iou > 0
                ORDER BY -iou.score, -iou.iou
            )

            LOOP
                INSERT INTO pairs (label_id, gtid, pdid, score, iou)
                VALUES (row.label_id, row.gt_id, row.pd_id, row.score, row.iou)
                ON CONFLICT DO NOTHING;
            END LOOP;

            RETURN QUERY SELECT pairs.label_id, pairs.gtid, pairs.pdid, pairs.score, pairs.iou FROM pairs;

            DROP TABLE pairs;

        END;
    $BODY$
    LANGUAGE plpgsql;
    """


def get_labels(dataset_id: int, annotationType: AnnotationType):
    """SQL query returns label table."""

    return f"""
    SELECT l.id, l.key, l.value
    FROM labeled_ground_truth_{annotationType} AS lgt
    JOIN label AS l
    ON lgt.label_id = l.id
    JOIN ground_truth_{annotationType} AS gt
    ON lgt.{annotationType}_id = gt.id
    JOIN datum
    ON gt.datum_id = datum.id
    WHERE dataset_id = {dataset_id}
    """


def get_number_of_ground_truths():
    """SQL query returns # of distinct ground truths grouped by label ID."""

    return """
    SELECT gt_label_id as label_id, count(distinct(gt_id))
    FROM iou
    WHERE gt_label_id IS NOT NULL
    GROUP BY gt_label_id
    """


def get_sorted_ranked_pairs():
    """SQL query returns ranked gt/pd pairs ordered by score, iou."""

    return """
    SELECT label_id, gt_id, pd_id, score, iou
    FROM find_ranked_pairs()
    ORDER BY -score, -iou
    """
