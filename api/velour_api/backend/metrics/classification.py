from typing import Sequence

import numpy as np
from sqlalchemy import Float, Integer
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, case, func, select

from velour_api import enums, schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    validate_computation,
)
from velour_api.backend.ops import Query

LabelMapType = list[list[list[str]]]


def _compute_binary_roc_auc(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label: schemas.Label,
) -> float:
    """
    Computes the binary ROC AUC score of a dataset and label.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    label : schemas.Label
        The label to compute the score for.

    Returns
    -------
    float
        The binary ROC AUC score.
    """
    # query to get the datum_ids and label values of groundtruths that have the given label key
    gts_filter = groundtruth_filter.model_copy()
    gts_filter.label_keys = [label.key]
    gts_query = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Label.value.label("label_value"),
        )
        .filter(gts_filter)
        .groundtruths("groundtruth_subquery")
    )

    # get the prediction scores for the given label (key and value)
    preds_filter = prediction_filter.model_copy()
    preds_filter.labels = [{label.key: label.value}]
    preds_query = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Prediction.score.label("score"),
            models.Label.value.label("label_value"),
        )
        .filter(preds_filter)
        .predictions("prediction_subquery")
    )

    # number of groundtruth labels that match the given label value
    n_pos = db.scalar(
        select(func.count(gts_query.c.label_value)).where(
            gts_query.c.label_value == label.value
        )
    )
    # total number of groundtruths
    n = db.scalar(select(func.count(gts_query.c.label_value)))

    if n is None or n_pos is None:
        raise RuntimeError(
            "ROCAUC computation failed; db.scalar returned None for mathematical variables."
        )

    if n_pos == 0:
        return 0

    if n - n_pos == 0:
        return 1.0

    # true positive rates
    tprs = (
        func.sum(
            (gts_query.c.label_value == label.value).cast(Integer).cast(Float)
        ).over(order_by=-preds_query.c.score)
        / n_pos
    )

    # false positive rates
    fprs = func.sum(
        (gts_query.c.label_value != label.value).cast(Integer).cast(Float)
    ).over(order_by=-preds_query.c.score) / (n - n_pos)

    tprs_fprs_query = (
        select(
            tprs.label("tprs"),
            fprs.label("fprs"),
            preds_query.c.score,
        ).join(
            preds_query,  # type: ignore - SQLAlchemy Subquery is incompatible with join type
            gts_query.c.datum_id == preds_query.c.datum_id,
        )
    ).subquery()

    trap_areas = select(
        (
            0.5
            * (
                tprs_fprs_query.c.tprs
                + func.lag(tprs_fprs_query.c.tprs).over(
                    order_by=-tprs_fprs_query.c.score
                )
            )
            * (
                tprs_fprs_query.c.fprs
                - func.lag(tprs_fprs_query.c.fprs).over(
                    order_by=-tprs_fprs_query.c.score
                )
            )
        ).label("trap_area")
    ).subquery()

    ret = db.scalar(func.sum(trap_areas.c.trap_area))
    if ret is None:
        return np.nan
    return ret


def _compute_roc_auc(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
) -> float | None:
    """
    Computes the area under the ROC curve. Note that for the multi-class setting
    this does one-vs-rest AUC for each class and then averages those scores. This should give
    the same thing as `sklearn.metrics.roc_auc_score` with `multi_class="ovr"`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_key : str
        The key of the grouper to calculate the ROCAUC for.
    grouper_mappings: dict[str, dict[str | int, any]]
        A dictionary of mappings that connect groupers to their related labels.

    Returns
    -------
    float | None
        The ROC AUC. Returns None if no labels exist for that label_key.
    """

    # get all of the labels associated with the grouper
    value_to_labels_mapping = grouper_mappings[
        "grouper_key_to_labels_mapping"
    ][grouper_key]

    sum_roc_aucs = 0
    label_count = 0

    for grouper_value, labels in value_to_labels_mapping.items():
        label_filter = groundtruth_filter.model_copy()
        label_filter.label_ids = [label.id for label in labels]

        # some labels in the "labels" argument may be out-of-scope given our groundtruth_filter, so we fetch all labels that are within scope of the groundtruth_filter to make sure we don't calculate ROCAUC for inappropriate labels
        in_scope_labels = [
            label
            for label in labels
            if schemas.Label(key=label.key, value=label.value)
            in core.get_labels(db=db, filters=label_filter)
        ]

        if not in_scope_labels:
            continue

        for label in labels:
            bin_roc = _compute_binary_roc_auc(
                db=db,
                prediction_filter=prediction_filter,
                groundtruth_filter=groundtruth_filter,
                label=schemas.Label(key=label.key, value=label.value),
            )

            if bin_roc is not None:
                sum_roc_aucs += bin_roc
                label_count += 1

    return sum_roc_aucs / label_count if label_count else None


def _compute_confusion_matrix_at_grouper_key(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
) -> schemas.ConfusionMatrix | None:
    """
    Computes the confusion matrix at a label_key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_mappings: dict[str, dict[str | int, any]]
        A dictionary of mappings that connect groupers to their related labels.

    Returns
    -------
    schemas.ConfusionMatrix | None
        Returns None in the case that there are no common images in the dataset
        that have both a groundtruth and prediction with label key `label_key`. Otherwise
        returns the confusion matrix.
    """
    label_key_filter = list(
        grouper_mappings["grouper_key_to_label_keys_mapping"][grouper_key]
    )

    # groundtruths filter
    gFilter = groundtruth_filter.model_copy()
    gFilter.label_keys = label_key_filter

    # predictions filter
    pFilter = prediction_filter.model_copy()
    pFilter.label_keys = label_key_filter

    # 0. Get groundtruths that conform to gFilter
    groundtruths = (
        Query(
            models.GroundTruth,
            models.Annotation.datum_id.label("datum_id"),
        )
        .filter(gFilter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    # 1. Get predictions that conform to pFilter
    predictions = (
        Query(models.Prediction)
        .filter(pFilter)
        .predictions(as_subquery=False)
        .alias()
    )

    # 2. Get the max prediction scores by datum
    max_scores_by_datum_id = (
        select(
            func.max(predictions.c.score).label("max_score"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 3. Remove duplicate scores per datum
    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    min_id_query = (
        select(
            func.min(predictions.c.id).label("min_id"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .select_from(predictions)
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .join(
            max_scores_by_datum_id,
            and_(
                models.Annotation.datum_id
                == max_scores_by_datum_id.c.datum_id,
                predictions.c.score == max_scores_by_datum_id.c.max_score,
            ),
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 4. Get labels for hard predictions, organize per datum
    hard_preds_query = (
        select(
            models.Label.value.label("pred_label_value"),
            min_id_query.c.datum_id.label("datum_id"),
        )
        .select_from(min_id_query)
        .join(
            models.Prediction,
            models.Prediction.id == min_id_query.c.min_id,
        )
        .join(
            models.Label,
            models.Label.id == models.Prediction.label_id,
        )
        .alias()
    )

    # 5. Link each label value to its corresponding grouper value
    b = Bundle(
        "cols",
        case(
            grouper_mappings["label_value_to_grouper_value"],
            value=hard_preds_query.c.pred_label_value,
        ),
        case(
            grouper_mappings["label_value_to_grouper_value"],
            value=models.Label.value,
        ),
    )

    # 6. Generate confusion matrix
    total_query = (
        select(b, func.count())
        .select_from(hard_preds_query)
        .join(
            groundtruths,
            groundtruths.c.datum_id == hard_preds_query.c.datum_id,
        )
        .join(
            models.Label,
            models.Label.id == groundtruths.c.label_id,
        )
        .group_by(b)  # type: ignore - SQLAlchemy Bundle not compatible with _first
    )

    res = db.execute(total_query).all()
    if len(res) == 0:
        # this means there's no predictions and groundtruths with the label key
        # for the same image
        return None

    return schemas.ConfusionMatrix(
        label_key=grouper_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
    )


def _compute_accuracy_from_cm(cm: schemas.ConfusionMatrix) -> float:
    """
    Computes the accuracy score from a confusion matrix.

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.

    Returns
    ----------
    float
        The resultant accuracy score.
    """
    return cm.matrix.trace() / cm.matrix.sum()


def _compute_precision_and_recall_f1_from_confusion_matrix(
    cm: schemas.ConfusionMatrix,
    label_value: str,
) -> tuple[float, float, float]:
    """
    Computes the precision, recall, and f1 score at a class index

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.
    label_key : str
        The label key to compute scores for.

    Returns
    ----------
    Tuple[float, float, float]
        A tuple containing the precision, recall, and F1 score.
    """
    cm_matrix = cm.matrix
    if label_value not in cm.label_map:
        return np.nan, np.nan, np.nan
    class_index = cm.label_map[label_value]

    true_positives = cm_matrix[class_index, class_index]
    # number of times the class was predicted
    n_preds = cm_matrix[:, class_index].sum()
    n_gts = cm_matrix[class_index, :].sum()

    prec = true_positives / n_preds if n_preds else 0
    recall = true_positives / n_gts if n_gts else 0

    f1_denom = prec + recall
    if f1_denom == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / f1_denom
    return prec, recall, f1


def _compute_confusion_matrix_and_metrics_at_grouper_key(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
) -> (
    tuple[
        schemas.ConfusionMatrix,
        list[
            schemas.AccuracyMetric
            | schemas.ROCAUCMetric
            | schemas.PrecisionMetric
            | schemas.RecallMetric
            | schemas.F1Metric
        ],
    ]
    | None
):
    """
    Computes the confusion matrix and all metrics for a given label key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_mappings: dict[str, dict[str | int, any]]
        A dictionary of mappings that connect groupers to their related labels.


    Returns
    -------
    tuple[schemas.ConfusionMatrix, list[schemas.AccuracyMetric | schemas.ROCAUCMetric | schemas.PrecisionMetric
                                        | schemas.RecallMetric | schemas.F1Metric]] | None
        Returns None if there are no predictions and groundtruths with the given label
        key for the same datum. Otherwise returns a tuple, with the first element the confusion
        matrix and the second a list of all metrics (accuracy, ROC AUC, precisions, recalls, and f1s).
    """

    confusion_matrix = _compute_confusion_matrix_at_grouper_key(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        grouper_key=grouper_key,
        grouper_mappings=grouper_mappings,
    )

    if confusion_matrix is None:
        return None

    # aggregate metrics (over all label values)
    metrics = [
        schemas.AccuracyMetric(
            label_key=grouper_key,
            value=_compute_accuracy_from_cm(confusion_matrix),
        ),
        schemas.ROCAUCMetric(
            label_key=grouper_key,
            value=_compute_roc_auc(
                db=db,
                prediction_filter=prediction_filter,
                groundtruth_filter=groundtruth_filter,
                grouper_key=grouper_key,
                grouper_mappings=grouper_mappings,
            ),
        ),
    ]

    # metrics that are per label
    for grouper_value in grouper_mappings["grouper_key_to_labels_mapping"][
        grouper_key
    ].keys():
        (
            precision,
            recall,
            f1,
        ) = _compute_precision_and_recall_f1_from_confusion_matrix(
            confusion_matrix, grouper_value
        )

        pydantic_label = schemas.Label(key=grouper_key, value=grouper_value)
        metrics.append(
            schemas.PrecisionMetric(
                label=pydantic_label,
                value=precision,
            )
        )
        metrics.append(
            schemas.RecallMetric(
                label=pydantic_label,
                value=recall,
            )
        )
        metrics.append(
            schemas.F1Metric(
                label=pydantic_label,
                value=f1,
            )
        )

    return confusion_matrix, metrics


def _compute_clf_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
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
    label_map: LabelMapType, optional
        Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.

    Returns
    ----------
    Tuple[List[schemas.ConfusionMatrix], List[schemas.ConfusionMatrix | schemas.AccuracyMetric | schemas.ROCAUCMetric| schemas.PrecisionMetric | schemas.RecallMetric | schemas.F1Metric]]
        A tuple of confusion matrices and metrics.
    """

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
        cm_and_metrics = _compute_confusion_matrix_and_metrics_at_grouper_key(
            db=db,
            prediction_filter=prediction_filter,
            groundtruth_filter=groundtruth_filter,
            grouper_key=grouper_key,
            grouper_mappings=grouper_mappings,
        )
        if cm_and_metrics is not None:
            confusion_matrices.append(cm_and_metrics[0])
            metrics.extend(cm_and_metrics[1])

    return confusion_matrices, metrics


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
    groundtruth_filter = schemas.Filter(**evaluation.datum_filter)
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    confusion_matrices, metrics = _compute_clf_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
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

    return evaluation_id
