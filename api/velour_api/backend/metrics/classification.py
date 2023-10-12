import numpy as np
from sqlalchemy import Float, Integer
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, func, select

from velour_api import schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.core import (
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.enums import EvaluationType, TaskType


# @TODO: Implement metadata filtering using `ops.BackendQuerys`
def binary_roc_auc(
    db: Session,
    dataset_name: str,
    model_name: str,
    label: schemas.Label,
) -> float:
    """Computes the binary ROC AUC score of a dataset and label

    Parameters
    ----------
    dataset_name
        name of the dataset
    model_name
        name of the model
    label
        the label that represents the positive class. all other labels with
        the same key as `label` will represent the negative class

    Returns
    -------
    float
        the binary ROC AUC score
    """

    # Retrieve sql objects
    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)

    # query to get the datum_ids and label values of groundtruths that have the given label key
    gts_query = (
        select(
            models.Annotation.datum_id.label("datum_id"),
            models.Label.value.label("label_value"),
        )
        .select_from(models.Label)
        .join(
            models.GroundTruth, models.GroundTruth.label_id == models.Label.id
        )
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id.is_(None),
                models.Annotation.task_type == TaskType.CLASSIFICATION.value,
                models.Label.key == label.key,
            ),
        )
    )
    gts_query = gts_query.subquery()

    # get the prediction scores for the given label (key and value)
    preds_query = (
        select(
            models.Annotation.datum_id.label("datum_id"),
            models.Prediction.score.label("score"),
            models.Label.value.label("label_value"),
        )
        .select_from(models.Label)
        .join(models.Prediction, models.Prediction.label_id == models.Label.id)
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id == model.id,
                models.Annotation.task_type == TaskType.CLASSIFICATION.value,
                models.Label.key == label.key,
                models.Label.value == label.value,
            ),
        )
    )
    preds_query = preds_query.subquery()

    # number of groundtruth labels that match the given label value
    n_pos = db.scalar(
        select(func.count(gts_query.c.label_value)).where(
            gts_query.c.label_value == label.value
        )
    )
    # total number of groundtruths
    n = db.scalar(select(func.count(gts_query.c.label_value)))

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
        ).join(preds_query, gts_query.c.datum_id == preds_query.c.datum_id)
        # .order_by(preds_query.c.score.desc())
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


# @TODO: Implement metadata filtering by using `ops.BackendQuery`
def roc_auc(
    db: Session,
    dataset_name: str,
    model_name: str,
    label_key: str,
    metadatum: list[schemas.Metadatum] = None,
) -> float:
    """Computes the area under the ROC curve. Note that for the multi-class setting
    this does one-vs-rest AUC for each class and then averages those scores. This should give
    the same thing as `sklearn.metrics.roc_auc_score` with `multi_class="ovr"`.

    Parameters
    ----------
    db
        database session
    dataset_name
        name of the dataset to
    label_key
        the label key to use
    metadata
        if not None, then filter out to just the datums that have this as a metadata

    Returns
    -------
    float
        ROC AUC
    """

    labels = {
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .join(
                models.GroundTruth,
                models.GroundTruth.label_id == models.Label.id,
            )
            .join(
                models.Annotation,
                models.Annotation.id == models.GroundTruth.annotation_id,
            )
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Dataset.id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Annotation.model_id.is_(None),
                    models.Label.key == label_key,
                )
            )
            .all()
        )
    }
    if len(labels) == 0:
        raise RuntimeError(
            f"The label key '{label_key}' is not a classification label in the dataset {dataset_name}."
        )

    sum_roc_aucs = 0
    label_count = 0
    for label in labels:
        bin_roc = binary_roc_auc(db, dataset_name, model_name, label)

        if bin_roc is not None:
            sum_roc_aucs += bin_roc
            label_count += 1

    return sum_roc_aucs / label_count


def get_confusion_matrix_and_metrics_at_label_key(
    db: Session,
    dataset_name: str,
    model_name: str,
    label_key: str,
    labels: list[models.Label],
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
    """Computes the confusion matrix and all metrics for a given label key

    Parameters
    ----------
    dataset_name
        name of the dataset
    model_name
        name of the model
    label_key
        the label key to compute metrics under
    labels
        all of the labels in both the groundtruths and predictions that have key
        equal to `label_key`

    Returns
    -------
    tuple[schemas.ConfusionMatrix, list[schemas.AccuracyMetric | schemas.ROCAUCMetric | schemas.PrecisionMetric
                                        | schemas.RecallMetric | schemas.F1Metric]] | None
        returns None if there are no predictions and groundtruths with the given label
        key for the same datum. Otherwise returns a tuple, with the first element the confusion
        matrix and the second a list of all metrics (accuracy, ROC AUC, precisions, recalls, and f1s)
    """

    for label in labels:
        if label.key != label_key:
            raise ValueError(
                f"Expected all elements of `labels` to have label key equal to {label_key} but got label {label}."
            )

    confusion_matrix = confusion_matrix_at_label_key(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        label_key=label_key,
    )

    if confusion_matrix is None:
        return None

    # aggregate metrics (over all label values)
    metrics = [
        schemas.AccuracyMetric(
            label_key=label_key,
            value=accuracy_from_cm(confusion_matrix),
            group=confusion_matrix.group,
        ),
        schemas.ROCAUCMetric(
            label_key=label_key,
            value=roc_auc(
                db,
                dataset_name,
                model_name,
                label_key,
            ),
            group=confusion_matrix.group,
        ),
    ]

    # metrics that are per label
    for label in labels:
        (
            precision,
            recall,
            f1,
        ) = precision_and_recall_f1_from_confusion_matrix(
            confusion_matrix, label.value
        )

        pydantic_label = schemas.Label(key=label.key, value=label.value)
        metrics.append(
            schemas.PrecisionMetric(
                label=pydantic_label,
                value=precision,
                group=confusion_matrix.group,
            )
        )
        metrics.append(
            schemas.RecallMetric(
                label=pydantic_label,
                value=recall,
                group=confusion_matrix.group,
            )
        )
        metrics.append(
            schemas.F1Metric(
                label=pydantic_label,
                value=f1,
                group=confusion_matrix.group,
            )
        )

    return confusion_matrix, metrics


def compute_clf_metrics(
    db: Session, dataset_name: str, model_name: str
) -> tuple[
    list[schemas.ConfusionMatrix],
    list[
        schemas.ConfusionMatrix
        | schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
    ],
]:
    ds_labels = {
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .select_from(models.Label)
            .join(
                models.GroundTruth,
                models.GroundTruth.label_id == models.Label.id,
            )
            .join(
                models.Annotation,
                models.Annotation.id == models.GroundTruth.annotation_id,
            )
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Annotation.task_type
                    == TaskType.CLASSIFICATION.value,
                    models.Annotation.model_id.is_(None),
                )
            )
            .all()
        )
    }
    md_labels = {
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .select_from(models.Label)
            .join(
                models.Prediction,
                models.Prediction.label_id == models.Label.id,
                full=True,
            )
            .join(
                models.Annotation,
                models.Annotation.id == models.Prediction.annotation_id,
            )
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
            .join(
                models.Model,
                models.Model.id == models.Annotation.model_id,
                full=True,
            )
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Annotation.task_type
                    == TaskType.CLASSIFICATION.value,
                    models.Annotation.model_id.isnot(None),
                    models.Model.name == model_name,
                )
            )
            .all()
        )
    }
    labels = list(ds_labels.union(md_labels))
    unique_label_keys = set([label.key for label in labels])

    confusion_matrices, metrics = [], []

    for label_key in unique_label_keys:

        def _add_confusion_matrix_and_metrics(**extra_kwargs):
            cm_and_metrics = get_confusion_matrix_and_metrics_at_label_key(
                db,
                dataset_name=dataset_name,
                model_name=model_name,
                label_key=label_key,
                labels=[label for label in labels if label.key == label_key],
                **extra_kwargs,
            )

            if cm_and_metrics is not None:
                confusion_matrices.append(cm_and_metrics[0])
                metrics.extend(cm_and_metrics[1])

        _add_confusion_matrix_and_metrics()

    return confusion_matrices, metrics


def confusion_matrix_at_label_key(
    db: Session,
    dataset_name: str,
    model_name: str,
    label_key: str,
    metadatum: schemas.Metadatum = None,
) -> schemas.ConfusionMatrix | None:
    """Computes the confusion matrix at a label_key.

    Parameters
    ----------
    dataset_name
        name of the dataset
    model_name
        name of the model
    label_key
        the label key to compute metrics under
    metadatum_id
        if not None, then filter out to just the datums that have this as a metadatum

    Returns
    -------
    schemas.ConfusionMatrix | None
        returns None in the case that there are no common images in the dataset
        that have both a groundtruth and prediction with label key `label_key`. Otherwise
        returns the confusion matrix
    """

    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)

    # this query get's the max score for each Datum for the given label key
    q1 = (
        select(
            func.max(models.Prediction.score).label("max_score"),
            models.Datum.id.label("datum_id"),
        )
        .select_from(models.Annotation)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(
            models.Prediction,
            models.Prediction.annotation_id == models.Annotation.id,
        )
        .join(models.Label, models.Label.id == models.Prediction.label_id)
        .where(
            and_(
                models.Annotation.model_id == model.id,
                models.Datum.dataset_id == dataset.id,
                models.Label.key == label_key,
            )
        )
    )
    if metadatum:
        string_value = (
            metadatum.value if isinstance(metadatum.value, str) else None
        )
        numeric_value = (
            metadatum.value if isinstance(metadatum.value, float) else None
        )
        q1 = q1.join(
            models.Metadatum, models.Metadatum.datum_id == models.Datum.id
        ).where(
            models.Metadatum.name == metadatum.name,
            models.Metadatum.string_value == string_value,
            models.Metadatum.numeric_value == numeric_value,
        )
    q1 = q1.group_by(models.Datum.id)
    subquery = q1.alias()

    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    q2 = (
        select(func.min(models.Prediction.id).label("min_id"))
        .join(models.Label)
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Datum, models.Annotation.datum_id == models.Datum.id)
        .join(
            subquery,
            and_(
                models.Prediction.score == subquery.c.max_score,
                models.Datum.id == subquery.c.datum_id,
            ),
        )
        .group_by(models.Datum.id)
    )
    min_id_query = q2.alias()

    q3 = (
        select(
            models.Label.value.label("pred_label_value"),
            models.Datum.id.label("datum_id"),
        )
        .join(models.Prediction)
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(
            subquery,
            and_(
                models.Prediction.score == subquery.c.max_score,
                models.Datum.id == subquery.c.datum_id,
            ),
        )
        .join(
            min_id_query,
            models.Prediction.id == min_id_query.c.min_id,
        )
    )
    hard_preds_query = q3.alias()

    b = Bundle("cols", hard_preds_query.c.pred_label_value, models.Label.value)

    total_query = (
        select(b, func.count())
        .join(
            models.Annotation,
            models.Annotation.datum_id == hard_preds_query.c.datum_id,
        )
        .join(
            models.GroundTruth,
            models.GroundTruth.annotation_id == models.Annotation.id,
        )
        .join(
            models.Label,
            and_(
                models.Label.id == models.GroundTruth.label_id,
                models.Label.key == label_key,
            ),
        )
        .group_by(b)
    )

    res = db.execute(total_query).all()

    if len(res) == 0:
        # this means there's no predictions and groundtruths with the label key
        # for the same image
        return None

    return schemas.ConfusionMatrix(
        label_key=label_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
        group=metadatum,
    )


def accuracy_from_cm(cm: schemas.ConfusionMatrix) -> float:
    return cm.matrix.trace() / cm.matrix.sum()


def precision_and_recall_f1_from_confusion_matrix(
    cm: schemas.ConfusionMatrix, label_value: str
) -> tuple[float, float, float]:
    """Computes the precision, recall, and f1 score at a class index"""
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


def create_clf_evaluation(
    db: Session,
    settings: schemas.EvaluationSettings,
) -> int:
    """This will always run in foreground.

    Returns
        Evaluations settings id.
    """

    dataset = core.get_dataset(db, settings.dataset)
    model = core.get_model(db, settings.model)

    es = get_or_create_row(
        db,
        models.Evaluation,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "type": EvaluationType.CLF,
            "constraints": None,
        },
    )

    return es.id


def create_clf_metrics(
    db: Session,
    settings: schemas.EvaluationSettings,
    evaluation_settings_id: int,
) -> int:
    """
    Intended to run as background
    """
    confusion_matrices, metrics = compute_clf_metrics(
        db=db,
        dataset_name=settings.dataset,
        model_name=settings.model,
    )

    confusion_matrices_mappings = create_metric_mappings(
        db=db,
        metrics=confusion_matrices,
        evaluation_settings_id=evaluation_settings_id,
    )

    for mapping in confusion_matrices_mappings:
        get_or_create_row(
            db,
            models.ConfusionMatrix,
            mapping,
        )

    metric_mappings = create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=evaluation_settings_id
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

    return evaluation_settings_id
