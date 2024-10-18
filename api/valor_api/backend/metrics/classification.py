import random
from collections import defaultdict

import numpy as np
from sqlalchemy import CTE, ColumnElement, Integer, literal
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, func, or_, select

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    commit_results,
    create_label_mapping,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_select


def _aggregate_data(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
) -> tuple[CTE, CTE, dict[int, tuple[str, str]]]:
    """
    Aggregates data for a classification task.

    This function returns a tuple containing CTE's used to gather groundtruths, predictions and a
    dictionary that maps label_id to a key-value pair.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    tuple[CTE, CTE, dict[int, tuple[str, str]]]:
        A tuple with form (groundtruths, predictions, labels).
    """
    labels = core.fetch_union_of_labels(
        db=db,
        lhs=groundtruth_filter,
        rhs=prediction_filter,
    )

    label_mapping = create_label_mapping(
        db=db,
        labels=labels,
        label_map=label_map,
    )

    groundtruths_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Label.id,
        label_mapping,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).subquery()
    groundtruths_cte = (
        select(
            groundtruths_subquery.c.datum_id,
            groundtruths_subquery.c.datum_uid,
            groundtruths_subquery.c.dataset_name,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
        )
        .select_from(groundtruths_subquery)
        .join(
            models.Label,
            models.Label.id == groundtruths_subquery.c.label_id,
        )
        .distinct()
        .cte()
    )

    predictions_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Model.name.label("model_name"),
        models.Prediction.id.label("prediction_id"),
        models.Prediction.score,
        models.Label.id,
        label_mapping,
        filters=prediction_filter,
        label_source=models.Prediction,
    ).subquery()
    predictions_cte = (
        select(
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.model_name,
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
            func.min(predictions_subquery.c.prediction_id).label(
                "prediction_id"
            ),
            func.max(predictions_subquery.c.score).label("score"),
        )
        .select_from(predictions_subquery)
        .join(
            models.Label,
            models.Label.id == predictions_subquery.c.label_id,
        )
        .group_by(
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.model_name,
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            models.Label.id,
            models.Label.key,
            models.Label.value,
        )
        .cte()
    )

    groundtruth_label_query = (
        db.query(
            groundtruths_cte.c.label_id,
            groundtruths_cte.c.key,
            groundtruths_cte.c.value,
        )
        .distinct()
        .all()
    )

    prediction_label_query = (
        db.query(
            predictions_cte.c.label_id,
            predictions_cte.c.key,
            predictions_cte.c.value,
        )
        .distinct()
        .all()
    )

    # get all labels
    groundtruth_labels = {
        label_id: (key, value)
        for label_id, key, value in groundtruth_label_query
    }
    prediction_labels = {
        label_id: (key, value)
        for label_id, key, value in prediction_label_query
    }
    labels = groundtruth_labels
    labels.update(prediction_labels)

    return (groundtruths_cte, predictions_cte, labels)


def _compute_clf_metrics(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    pr_curve_max_examples: int,
    metrics_to_return: list[enums.MetricType],
    label_map: LabelMapType | None = None,
) -> list[
    schemas.ConfusionMatrix
    | schemas.AccuracyMetric
    | schemas.ROCAUCMetric
    | schemas.PrecisionMetric
    | schemas.RecallMetric
    | schemas.F1Metric
    | schemas.PrecisionRecallCurve
    | schemas.DetailedPrecisionRecallCurve
]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.

    Returns
    ----------
    list[ConfusionMatrix, Metric]
        A list of confusion matrices and metrics.
    """

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=label_map,
    )

    # compute metrics and confusion matrix for each grouper id
    confusion_matrices, metrics = _compute_confusion_matrices_and_metrics(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=labels,
        pr_curve_max_examples=pr_curve_max_examples,
        metrics_to_return=metrics_to_return,
    )

    return confusion_matrices + metrics


@validate_computation
def compute_clf_metrics(
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

    # fetch evaluation
    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    # unpack filters and params
    parameters = schemas.EvaluationParameters(**evaluation.parameters)
    groundtruth_filter, prediction_filter = prepare_filter_for_evaluation(
        filters=schemas.Filter(**evaluation.filters),
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
    )

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    if parameters.metrics_to_return is None:
        raise RuntimeError("Metrics to return should always be defined here.")

    metrics = _compute_clf_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
        pr_curve_max_examples=(
            parameters.pr_curve_max_examples
            if parameters.pr_curve_max_examples
            else 0
        ),
        metrics_to_return=parameters.metrics_to_return,
    )

    # add metrics to database
    commit_results(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation.id,
    )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
