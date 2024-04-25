from sqlalchemy.orm import Session
from sqlalchemy.sql import case, func, select

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
    compute_pr_curves: bool
        A boolean which determines whether we calculate precision-recall curves or not.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

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
                grouper_mappings["label_key_to_grouper_key"],
                value=models.Label.key,
            ).label("grouper_key"),
            models.Label.value.label("gt_value"),
            models.Annotation.ranking.label("gt_ranking"),
        )
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    predictions = (
        Query(
            models.Prediction,
            case(
                grouper_mappings["label_key_to_grouper_key"],
                value=models.Label.key,
            ).label("grouper_key"),
            models.Label.value.label("pd_value"),
            models.Annotation.ranking.label("pd_ranking"),
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
        .alias()
    )

    # blah_rr = db.execute(predictions).all()
    # import pdb

    # pdb.set_trace()

    joint = (
        select(
            groundtruths.c.grouper_key,
            groundtruths.c.gt_ranking,
            predictions.c.pd_ranking,
        )
        .select_from(groundtruths)
        .join(
            predictions,
            groundtruths.c.grouper_key == predictions.c.grouper_key,
        )
    ).alias()

    flattened_predictions = select(
        joint.c.grouper_key,
        func.jsonb_array_elements_text(joint.c.pd_ranking).label("pd_rank"),
    ).select_from(joint)

    pd_rankings = select(
        flattened_predictions.c.grouper_key,
        flattened_predictions.c.pd_rank,
        func.row_number()
        .over(partition_by=flattened_predictions.c.grouper_key)
        .label("row_number"),
    ).alias()

    flattened_groundtruths = (
        select(
            joint.c.grouper_key,
            func.jsonb_array_elements_text(joint.c.gt_ranking).label(
                "gt_rank"
            ),
        )
        .select_from(joint)
        .alias()
    )

    reciprocal_rankings = (
        select(
            flattened_groundtruths.c.grouper_key,
            func.coalesce(func.min(pd_rankings.c.row_number), 0).label("rr"),
        )
        .select_from(flattened_groundtruths)
        .join(
            pd_rankings,
            pd_rankings.c.grouper_key == flattened_groundtruths.c.grouper_key,
        )
        .group_by(flattened_groundtruths.c.grouper_key)
        .alias()
    )

    mean_reciprocal_ranks = (
        select(
            reciprocal_rankings.c.grouper_key,
            func.avg(reciprocal_rankings.c.rr),
        )
        .select_from(reciprocal_rankings)
        .group_by(reciprocal_rankings.c.grouper_key)
    )

    mrrs_by_key = db.execute(mean_reciprocal_ranks).all()

    for grouper_key, mrr in mrrs_by_key:
        metrics.append(schemas.MRRMetric(label_key=grouper_key, value=mrr))

    # TODO unit tests for MRRMetrics schema

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
