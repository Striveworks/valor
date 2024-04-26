# from collections import defaultdict
from typing import Sequence

# import numpy as np
# from sqlalchemy import Integer, Subquery
from sqlalchemy.orm import Session  # , Bundle

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

# from sqlalchemy.sql import and_, case, func, select
# from sqlalchemy.sql.selectable import NamedFromClause

# from valor_api.backend.query import Query

LabelMapType = list[list[list[str]]]


def _compute_llm_evaluation_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    compute_pr_curves: bool,
    label_map: LabelMapType | None = None,
) -> tuple[
    list[schemas.ConfusionMatrix],
    Sequence[
        schemas.ConfusionMatrix
        | schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
    ],
]:
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
    raise NotImplementedError

    labels = core.fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=label_map,
        evaluation_type=enums.TaskType.CLASSIFICATION,
    )

    # compute metrics and confusion matrix for each grouper id
    confusion_matrices, metrics = [], []
    for grouper_key in grouper_mappings[
        "grouper_key_to_labels_mapping"
    ].keys():
        cm_and_metrics = None  # this was not in the original code
        # cm_and_metrics = _compute_confusion_matrix_and_metrics_at_grouper_key(
        #     db=db,
        #     prediction_filter=prediction_filter,
        #     groundtruth_filter=groundtruth_filter,
        #     grouper_key=grouper_key,
        #     grouper_mappings=grouper_mappings,
        #     compute_pr_curves=compute_pr_curves,
        # )
        if cm_and_metrics is not None:
            confusion_matrices.append(cm_and_metrics[0])
            metrics.extend(cm_and_metrics[1])

    return confusion_matrices, metrics


@validate_computation
def compute_llm_evaluation_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create classification metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

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

    raise NotImplementedError

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

    confusion_matrices, metrics = _compute_llm_evaluation_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
        compute_pr_curves=(
            parameters.compute_pr_curves
            if parameters.compute_pr_curves is not None
            else False
        ),
    )

    confusion_matrices_mappings = create_metric_mappings(
        db=db,
        metrics=confusion_matrices,
        evaluation_id=evaluation.id,
    )

    for mapping in confusion_matrices_mappings:
        get_or_create_row(
            db,
            models.ConfusionMatrix,
            mapping,
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
