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

# TODO consolidate this?
LabelMapType = list[list[list[str]]]


def _compute_ranking_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
    k_cutoffs: list[int] = [1, 3, 5],
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
            models.GroundTruth,
            case(
                grouper_mappings["label_to_grouper_key"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_key"),
            case(
                grouper_mappings["label_to_grouper_value"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_value"),
            models.Annotation.ranking.label("gt_ranking"),
            models.Annotation.datum_id,
            models.Annotation.id.label("gt_annotation_id"),
        )
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    predictions = (
        Query(
            models.Prediction,  # TODO is this deletable?
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

    # needed to avoid a type error when joining
    aliased_rr = reciprocal_rankings.alias()

    # add back any predictions that didn't contain any relevant elements
    gts_with_no_predictions = select(
        joint.c.grouper_key,
        joint.c.grouper_value,
        joint.c.gt_annotation_id.label("annotation_id"),
        literal(0).label("recip_rank"),
    ).where(joint.c.pd_annotation_id.is_(None))

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
            aliased_rr,
            and_(
                pd_rankings.c.grouper_key == aliased_rr.c.grouper_key,
                pd_rankings.c.grouper_value == aliased_rr.c.grouper_value,
                pd_rankings.c.pd_annotation_id == aliased_rr.c.annotation_id,
            ),
            isouter=True,
        )
        .where(aliased_rr.c.annotation_id.is_(None))
    )

    aliased_predictions_with_no_gts = predictions_with_no_gts.alias()

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

    for grouper_key, mrr in mrrs_by_key:
        metrics.append(schemas.MRRMetric(label_key=grouper_key, value=mrr))

    # calculate precision @k to measure how many items with the top k positions are relevant.
    sum_functions = [
        func.sum(
            case(
                (filtered_rankings.c.rank <= k, 1),
                else_=0,
            )
        ).label(f"precision@{k}")
        for k in k_cutoffs
    ]

    # calculate precision@k
    calc_precision_at_k = (
        select(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id,
            *sum_functions,
        )
        .select_from(filtered_rankings)
        .group_by(
            filtered_rankings.c.grouper_key,
            filtered_rankings.c.grouper_value,
            filtered_rankings.c.pd_annotation_id,
        )
        .union_all(  # add back predictions that don't have any groundtruths
            select(
                predictions_with_no_gts.c.grouper_key,
                predictions_with_no_gts.c.grouper_value,
                predictions_with_no_gts.c.annotation_id,
                *(literal(0).label(f"precision@{k}") for k in k_cutoffs),
            ).select_from(aliased_predictions_with_no_gts)
        )
    ).alias()

    # roll up precision@k into ap@k
    columns = [calc_precision_at_k.c[f"precision@{k}"] for k in k_cutoffs]

    calc_ap_at_k = select(
        calc_precision_at_k.c.grouper_key,
        calc_precision_at_k.c.grouper_value,
        calc_precision_at_k.c.pd_annotation_id,
        func.sum(sum(columns) / len(columns)).label("ap@k"),
    ).group_by(
        calc_precision_at_k.c.grouper_key,
        calc_precision_at_k.c.grouper_value,
        calc_precision_at_k.c.pd_annotation_id,
    )

    # roll up ap@k into map@k
    calc_map_at_k = select(
        calc_ap_at_k.c.grouper_key,
        func.avg(calc_ap_at_k.c["ap@k"]).label("map@k"),
    ).group_by(
        calc_ap_at_k.c.grouper_key,
    )

    # execute queries
    precision_at_k = db.query(calc_precision_at_k).all()
    ap_at_k = db.execute(calc_ap_at_k).all()
    map_at_k = db.execute(calc_map_at_k).all()

    for metric in precision_at_k:
        key, value, annotation_id, precisions = (
            metric[0],
            metric[1],
            metric[2],
            metric[3:],
        )

        for i, k in enumerate(k_cutoffs):
            metrics.append(
                schemas.PrecisionAtKMetric(
                    label=schemas.Label(key=key, value=value),
                    value=precisions[i],
                    k=k,
                    annotation_id=annotation_id,
                )
            )

    for metric in ap_at_k:
        key, value, annotation_id, metric_value = (
            metric[0],
            metric[1],
            metric[2],
            metric[3],
        )

        metrics.append(
            schemas.APAtKMetric(
                label=schemas.Label(key=key, value=value),
                value=metric_value,
                k_cutoffs=k_cutoffs,
                annotation_id=annotation_id,
            )
        )

    for metric in map_at_k:
        key, metric_value = (
            metric[0],
            metric[1],
        )

        metrics.append(
            schemas.mAPAtKMetric(
                label_key=key,
                value=metric_value,
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
