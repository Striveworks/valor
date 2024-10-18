from collections import defaultdict
from typing import Any

from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy import CTE, Subquery, and_, case, func, select
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    LabelMapType,
    commit_results,
    create_label_mapping,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_select
from valor_api.schemas.metrics import (
    F1Metric,
    IOUMetric,
    PrecisionMetric,
    RecallMetric,
    mIOUMetric,
)


def _aggregate_data(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
) -> tuple[CTE, CTE, dict[int, tuple[str, str]]]:
    """
    Aggregates data for a semantic segmentation task.

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
        models.Dataset.name.label("dataset_name"),
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Annotation.id.label("annotation_id"),
        models.Annotation.raster.label("raster"),
        models.GroundTruth.id.label("groundtruth_id"),
        models.Label.id,
        label_mapping,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).subquery()
    groundtruths_cte = (
        select(
            groundtruths_subquery.c.dataset_name,
            groundtruths_subquery.c.datum_id,
            groundtruths_subquery.c.datum_uid,
            groundtruths_subquery.c.annotation_id,
            groundtruths_subquery.c.groundtruth_id,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
            func.ST_Union(groundtruths_subquery.c.raster).label("raster"),
        )
        .select_from(groundtruths_subquery)
        .join(
            models.Label,
            models.Label.id == groundtruths_subquery.c.label_id,
        )
        .group_by(
            groundtruths_subquery.c.dataset_name,
            groundtruths_subquery.c.datum_id,
            groundtruths_subquery.c.datum_uid,
            groundtruths_subquery.c.annotation_id,
            groundtruths_subquery.c.groundtruth_id,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
        )
        .cte()
    )

    predictions_subquery = generate_select(
        models.Dataset.name.label("dataset_name"),
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Annotation.id.label("annotation_id"),
        models.Annotation.raster.label("raster"),
        models.Prediction.id.label("prediction_id"),
        models.Prediction.score.label("score"),
        models.Label.id,
        label_mapping,
        filters=prediction_filter,
        label_source=models.Prediction,
    ).subquery()
    predictions_cte = (
        select(
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.annotation_id,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
            func.min(predictions_subquery.c.prediction_id).label(
                "prediction_id"
            ),
            func.max(predictions_subquery.c.score).label("score"),
            func.ST_UNION(predictions_subquery.c.raster).label("raster"),
        )
        .select_from(predictions_subquery)
        .join(
            models.Label,
            models.Label.id == predictions_subquery.c.label_id,
        )
        .group_by(
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.annotation_id,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
        )
        .cte()
    )

    # get all labels
    groundtruth_labels = {
        (key, value, label_id)
        for label_id, key, value in db.query(
            groundtruths_cte.c.label_id,
            groundtruths_cte.c.key,
            groundtruths_cte.c.value,
        )
        .distinct()
        .all()
    }
    prediction_labels = {
        (key, value, label_id)
        for label_id, key, value in db.query(
            predictions_cte.c.label_id,
            predictions_cte.c.key,
            predictions_cte.c.value,
        )
        .distinct()
        .all()
    }
    labels = groundtruth_labels.union(prediction_labels)
    labels = {label_id: (key, value) for key, value, label_id in labels}

    return (groundtruths_cte, predictions_cte, labels)


def _compute_segmentation_metrics(
    db: Session,
    parameters: schemas.EvaluationParameters,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
) -> list[IOUMetric | mIOUMetric]:
    """
    Computes segmentation metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    parameters : schemas.EvaluationParameters
        Any user-defined parameters.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.

    Returns
    ----------
    List[schemas.IOUMetric | mIOUMetric | PrecisionMetric | RecallMetric | F1Metric | AccuracyMetric]:
        A list containing one `IOUMetric` for each label in ground truth and one `mIOUMetric` for the mean _compute_IOU over all labels.
    """

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=parameters.label_map,
    )

    return _compute_iou(
        db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=labels,
    )


@validate_computation
def compute_semantic_segmentation_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create semantic segmentation metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.
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

    metrics = _compute_segmentation_metrics(
        db=db,
        parameters=parameters,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    # add metrics to database
    commit_results(db, metrics, evaluation_id)

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
