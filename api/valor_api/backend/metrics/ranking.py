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
                grouper_mappings["label_to_grouper_key"],
                value=func.concat(models.Label.key, models.Label.value),
            ).label("grouper_id"),
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
            ).label("grouper_id"),
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
            groundtruths.c.grouper_id,
            groundtruths.c.gt_ranking,
            groundtruths.c.gt_annotation_id,
            predictions.c.pd_ranking,
            predictions.c.pd_annotation_id,
        )
        .select_from(groundtruths)
        .join(
            predictions,
            and_(
                groundtruths.c.grouper_id == predictions.c.grouper_id,
                groundtruths.c.datum_id == predictions.c.datum_id,
            ),
            isouter=True,  # left join to include ground truths without predictions
        )
    )

    flattened_predictions = select(
        joint.c.grouper_id,
        joint.c.pd_annotation_id,
        func.jsonb_array_elements_text(joint.c.pd_ranking).label("pd_item"),
    )

    flattened_groundtruths = select(
        joint.c.grouper_id,
        func.jsonb_array_elements_text(joint.c.gt_ranking).label("gt_item"),
    ).alias()

    pd_rankings = select(
        flattened_predictions.c.grouper_id,
        flattened_predictions.c.pd_annotation_id,
        flattened_predictions.c.pd_item,
        func.row_number()
        .over(
            partition_by=and_(
                flattened_predictions.c.pd_annotation_id,
                flattened_predictions.c.grouper_id,
            )
        )
        .label("rank"),
    ).alias()

    # filter pd_rankings to only include relevant docs, then find the reciprical rank
    reciprocal_rankings = (
        select(
            pd_rankings.c.grouper_id,
            pd_rankings.c.pd_annotation_id.label("annotation_id"),
            1 / func.min(pd_rankings.c.rank).label("recip_rank"),
        )
        .distinct()
        .select_from(pd_rankings)
        .join(
            flattened_groundtruths,
            and_(
                flattened_groundtruths.c.grouper_id
                == pd_rankings.c.grouper_id,
                flattened_groundtruths.c.gt_item == pd_rankings.c.pd_item,
            ),
        )
        .group_by(
            pd_rankings.c.grouper_id,
            pd_rankings.c.pd_annotation_id,
        )
    )

    # needed to avoid a type error when joining
    aliased_rr = reciprocal_rankings.alias()

    # add back any predictions that didn't contain any relevant elements
    gts_with_no_predictions = select(
        joint.c.grouper_id,
        joint.c.gt_annotation_id.label("annotation_id"),
        literal(0).label("recip_rank"),
    ).where(joint.c.pd_annotation_id.is_(None))

    rrs_with_missing_entries = (
        select(
            pd_rankings.c.grouper_id,
            pd_rankings.c.pd_annotation_id.label("annotation_id"),
            literal(0).label("recip_rank"),
        )
        .distinct()
        .select_from(pd_rankings)
        .join(
            aliased_rr,
            and_(
                pd_rankings.c.grouper_id == aliased_rr.c.grouper_id,
                pd_rankings.c.pd_annotation_id == aliased_rr.c.annotation_id,
            ),
            isouter=True,
        )
        .where(aliased_rr.c.annotation_id.is_(None))
    ).union_all(reciprocal_rankings, gts_with_no_predictions)

    # take the mean across grouper keys to arrive at the MRR per key
    mrrs_by_key = db.execute(
        select(
            rrs_with_missing_entries.c.grouper_id,
            func.avg(rrs_with_missing_entries.c.recip_rank).label("mrr"),
        ).group_by(rrs_with_missing_entries.c.grouper_id)
    ).all()

    for grouper_id, mrr in mrrs_by_key:
        metrics.append(schemas.MRRMetric(label_key=grouper_id, value=mrr))

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
