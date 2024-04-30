from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, case, func, literal, select

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    validate_computation,
)
from valor_api.backend.query import Query

LabelMapType = list[list[list[str]]]


def _calc_precision_recall_at_k_metrics(
    db: Session, k_cutoffs, filtered_rankings, predictions_with_no_gts
):
    """Calculate all metrics related to precision@k and recall@k."""
    precision_and_recall_functions = [
        (
            func.sum(
                case(
                    (filtered_rankings.c.rank <= k, 1),
                    else_=0,
                )
            )
            / k
        ).label(f"precision@{k}")
        for k in k_cutoffs
    ]

    recall_at_k_functions = [
        (
            func.sum(
                case(
                    (filtered_rankings.c.rank <= k, 1),
                    else_=0,
                )
            )
            / filtered_rankings.c.number_of_relevant_items
        ).label(f"recall@{k}")
        for k in k_cutoffs
    ]

    calc_recall_and_precision = (
        select(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id,
            *precision_and_recall_functions,
            *recall_at_k_functions,
        )
        .select_from(filtered_rankings)
        .group_by(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id,
            filtered_rankings.c.number_of_relevant_items,
        )
        .union_all(  # add back predictions that don't have any groundtruths
            select(
                predictions_with_no_gts.c.grouper_key,
                predictions_with_no_gts.c.grouper_value,
                predictions_with_no_gts.c.annotation_id,
                *(literal(0).label(f"precision@{k}") for k in k_cutoffs),
                *(literal(0).label(f"recall@{k}") for k in k_cutoffs),
            ).select_from(predictions_with_no_gts)
        )
    ).alias()

    # roll up by k
    precision_columns = [
        calc_recall_and_precision.c[f"precision@{k}"] for k in k_cutoffs
    ]
    recall_columns = [
        calc_recall_and_precision.c[f"recall@{k}"] for k in k_cutoffs
    ]

    calc_ap_and_ar = select(
        calc_recall_and_precision.c.grouper_key,
        calc_recall_and_precision.c.grouper_value,
        calc_recall_and_precision.c.pd_annotation_id,
        func.sum(sum(precision_columns) / len(precision_columns)).label(
            "ap@k"
        ),
        func.sum(sum(recall_columns) / len(recall_columns)).label("ar@k"),
    ).group_by(
        calc_recall_and_precision.c.grouper_key,
        calc_recall_and_precision.c.grouper_value,
        calc_recall_and_precision.c.pd_annotation_id,
    )

    # roll up by label key
    calc_map_and_mar = select(
        calc_ap_and_ar.c.grouper_key,
        func.avg(calc_ap_and_ar.c["ap@k"]).label("map@k"),
        func.avg(calc_ap_and_ar.c["ar@k"]).label("mar@k"),
    ).group_by(
        calc_ap_and_ar.c.grouper_key,
    )

    # execute queries
    precision_and_recall = db.query(calc_recall_and_precision).all()
    ap_and_ar = db.execute(calc_ap_and_ar).all()
    map_and_mar = db.execute(calc_map_and_mar).all()

    return precision_and_recall, ap_and_ar, map_and_mar


def _calc_mrr(db: Session, joint, filtered_rankings, predictions_with_no_gts):
    """Calculates mean reciprical rank (MRR) metrics."""
    # calculate reciprocal rankings for MRR
    reciprocal_rankings = (
        select(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id.label("annotation_id"),
            1 / func.min(filtered_rankings.c.rank).label("recip_rank"),
        )
        .select_from(filtered_rankings)
        .group_by(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id,
        )
    )

    # add back any predictions that didn't contain any relevant elements
    gts_with_no_predictions = select(
        joint.c.grouper_key,
        joint.c.grouper_value,
        joint.c.gt_annotation_id.label("annotation_id"),
        literal(0).label("recip_rank"),
    ).where(joint.c.pd_annotation_id.is_(None))

    mrr_union = predictions_with_no_gts.union_all(
        reciprocal_rankings, gts_with_no_predictions
    )

    # take the mean across grouper keys to arrive at the MRR per key
    mrrs_by_key = db.execute(
        select(
            mrr_union.c.grouper_key,
            func.avg(mrr_union.c.recip_rank).label("mrr"),
        ).group_by(mrr_union.c.grouper_key)
    ).all()

    return mrrs_by_key


def _compute_ranking_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
    k_cutoffs: list[int] = [1, 3, 5],
    metrics_to_return: list[str] = [
        "MRRMetric",
        "PrecisionAtKMetric",
        "APAtKMetric",
        "mAPAtKMetric",
        "RecallAtKMetric",
        "ARAtKMetric",
        "mARAtKMetric",
        "MRRMetric",
    ],
) -> list:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    k_cutoffs: list[int]
        The cut-offs (or "k" values) to use when calculating precision@k, recall@k, etc.
    metrics_to_return: list[str]
        The list of metrics to return to the user.

    Returns
    ----------
    Tuple[List[schemas.ConfusionMatrix], List[schemas.ConfusionMatrix | schemas.AccuracyMetric | schemas.ROCAUCMetric| schemas.PrecisionMetric | schemas.RecallMetric | schemas.F1Metric]]
        A tuple of confusion matrices and metrics.
    """

    metrics = []

    labels = core.fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=label_map,
        evaluation_type=enums.TaskType.RANKING,
    )

    metrics = []

    groundtruths = (
        Query(
            case(
                grouper_mappings["label_to_grouper_key"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_key"),
            case(
                grouper_mappings["label_to_grouper_value"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_value"),
            models.Annotation.ranking.label("gt_ranking"),
            func.jsonb_array_length(models.Annotation.ranking).label(
                "number_of_relevant_items"
            ),
            models.Annotation.datum_id,
            models.Annotation.id.label("gt_annotation_id"),
        )
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    predictions = (
        Query(
            case(
                grouper_mappings["label_to_grouper_key"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_key"),
            case(
                grouper_mappings["label_to_grouper_value"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_value"),
            models.Annotation.ranking.label("pd_ranking"),
            models.Annotation.datum_id,
            models.Annotation.id.label("pd_annotation_id"),
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
        .alias()
    )

    joint = (
        select(
            groundtruths.c.grouper_key,
            groundtruths.c.grouper_value,
            groundtruths.c.gt_ranking,
            groundtruths.c.gt_annotation_id,
            groundtruths.c.number_of_relevant_items,
            predictions.c.pd_ranking,
            predictions.c.pd_annotation_id,
        )
        .select_from(groundtruths)
        .join(
            predictions,
            and_(
                groundtruths.c.grouper_key == predictions.c.grouper_key,
                groundtruths.c.grouper_value == predictions.c.grouper_value,
                groundtruths.c.datum_id == predictions.c.datum_id,
            ),
            isouter=True,  # left join to include ground truths without predictions
        )
    )

    flattened_predictions = select(
        joint.c.grouper_key,
        joint.c.grouper_value,
        joint.c.pd_annotation_id,
        func.jsonb_array_elements_text(joint.c.pd_ranking).label("pd_item"),
    )

    flattened_groundtruths = select(
        joint.c.grouper_key,
        joint.c.grouper_value,
        joint.c.number_of_relevant_items,
        func.jsonb_array_elements_text(joint.c.gt_ranking).label("gt_item"),
    ).alias()

    pd_rankings = select(
        flattened_predictions.c.grouper_key,
        flattened_predictions.c.grouper_value,
        flattened_predictions.c.pd_annotation_id,
        flattened_predictions.c.pd_item,
        func.row_number()
        .over(
            partition_by=and_(
                flattened_predictions.c.pd_annotation_id,
                flattened_predictions.c.grouper_key,
                flattened_predictions.c.grouper_value,
            )
        )
        .label("rank"),
    ).alias()

    # filter pd_rankings to only include relevant docs, then find the reciprical rank
    filtered_rankings = (
        select(
            pd_rankings.c.grouper_key,
            pd_rankings.c.grouper_value,
            pd_rankings.c.pd_annotation_id,
            pd_rankings.c.rank,
            flattened_groundtruths.c.number_of_relevant_items,
        )
        .distinct()
        .select_from(pd_rankings)
        .join(
            flattened_groundtruths,
            and_(
                flattened_groundtruths.c.grouper_key
                == pd_rankings.c.grouper_key,
                flattened_groundtruths.c.grouper_value
                == pd_rankings.c.grouper_value,
                flattened_groundtruths.c.gt_item == pd_rankings.c.pd_item,
            ),
        )
    ).alias()

    predictions_with_no_gts = (
        select(
            pd_rankings.c.grouper_key,
            pd_rankings.c.grouper_value,
            pd_rankings.c.pd_annotation_id.label("annotation_id"),
            literal(0).label("recip_rank"),
        )
        .distinct()
        .select_from(pd_rankings)
        .join(
            filtered_rankings,
            and_(
                pd_rankings.c.grouper_key == filtered_rankings.c.grouper_key,
                pd_rankings.c.grouper_value
                == filtered_rankings.c.grouper_value,
                pd_rankings.c.pd_annotation_id
                == filtered_rankings.c.pd_annotation_id,
            ),
            isouter=True,
        )
        .where(filtered_rankings.c.pd_annotation_id.is_(None))
    )

    aliased_predictions_with_no_gts = predictions_with_no_gts.alias()

    if "MRRMetric" in metrics_to_return:
        mrrs_by_key = _calc_mrr(
            db=db,
            joint=joint,
            filtered_rankings=filtered_rankings,
            predictions_with_no_gts=predictions_with_no_gts,
        )

        for grouper_key, mrr in mrrs_by_key:
            metrics.append(schemas.MRRMetric(label_key=grouper_key, value=mrr))

    if not set(metrics_to_return).isdisjoint(
        set(
            [
                "PrecisionAtKMetric",
                "APAtKMetric",
                "mAPAtKMetric",
                "RecallAtKMetric",
                "ARAtKMetric",
                "mARAtKMetric",
                "MRRMetric",
            ]
        )
    ):

        (
            precision_and_recall,
            ap_and_ar,
            map_and_mar,
        ) = _calc_precision_recall_at_k_metrics(
            db=db,
            k_cutoffs=k_cutoffs,
            filtered_rankings=filtered_rankings,
            predictions_with_no_gts=aliased_predictions_with_no_gts,
        )

        for metric in precision_and_recall:
            key, value, annotation_id, precisions, recalls = (
                metric[0],
                metric[1],
                metric[2],
                metric[3 : 3 + len(k_cutoffs)],
                metric[3 + len(k_cutoffs) :],
            )

            for i, k in enumerate(k_cutoffs):
                if "PrecisionAtKMetric" in metrics_to_return:
                    metrics.append(
                        schemas.PrecisionAtKMetric(
                            label=schemas.Label(key=key, value=value),
                            value=precisions[i],
                            k=k,
                            annotation_id=annotation_id,
                        )
                    )

                if "RecallAtKMetric" in metrics_to_return:
                    metrics.append(
                        schemas.RecallAtKMetric(
                            label=schemas.Label(key=key, value=value),
                            value=recalls[i],
                            k=k,
                            annotation_id=annotation_id,
                        )
                    )

        for metric in ap_and_ar:
            key, value, annotation_id, ap_value, ar_value = (
                metric[0],
                metric[1],
                metric[2],
                metric[3],
                metric[4],
            )

            if "APAtKMetric" in metrics_to_return:
                metrics.append(
                    schemas.APAtKMetric(
                        label=schemas.Label(key=key, value=value),
                        value=ap_value,
                        k_cutoffs=k_cutoffs,
                        annotation_id=annotation_id,
                    )
                )

            if "ARAtKMetric" in metrics_to_return:
                metrics.append(
                    schemas.ARAtKMetric(
                        label=schemas.Label(key=key, value=value),
                        value=ar_value,
                        k_cutoffs=k_cutoffs,
                        annotation_id=annotation_id,
                    )
                )

        for metric in map_and_mar:
            key, map_value, mar_value = (metric[0], metric[1], metric[2])

            if "mAPAtKMetric" in metrics_to_return:
                metrics.append(
                    schemas.mAPAtKMetric(
                        label_key=key,
                        value=map_value,
                        k_cutoffs=k_cutoffs,
                    )
                )

            if "mARAtKMetric" in metrics_to_return:
                metrics.append(
                    schemas.mARAtKMetric(
                        label_key=key,
                        value=mar_value,
                        k_cutoffs=k_cutoffs,
                    )
                )

    return metrics


@validate_computation
def compute_ranking_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create ranking metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.

    Returns
    ----------
    int
        The evaluation job id.
    """

    # fetch evaluation
    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    # unpack filters and params
    groundtruth_filter = schemas.Filter(**evaluation.datum_filter)
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    # if the user doesn't specify any metrics, then return all of them
    if parameters.metrics_to_return is None:
        parameters.metrics_to_return = [
            "MRRMetric",
            "PrecisionAtKMetric",
            "APAtKMetric",
            "mAPAtKMetric",
            "RecallAtKMetric",
            "ARAtKMetric",
            "mARAtKMetric",
            "MRRMetric",
        ]

    # if the user doesn't specify any k-cutoffs, then use [1, 3, 5] as the default
    if parameters.k_cutoffs is None:
        parameters.k_cutoffs = [1, 3, 5]

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    metrics = _compute_ranking_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
        metrics_to_return=parameters.metrics_to_return,
        k_cutoffs=parameters.k_cutoffs,
    )

    metric_mappings = create_metric_mappings(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation.id,
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        get_or_create_row(
            db,
            models.Metric,
            mapping,
            columns_to_ignore=["value"],
        )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
