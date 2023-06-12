import numpy as np
from sqlalchemy import Float, Integer
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, func, select

from velour_api import crud, models, schemas
from velour_api.models import PredictedClassification


def binary_roc_auc(
    db: Session, dataset_name: str, model_name: str, label: schemas.Label
) -> float:
    # query to get the datum_ids and label values of groundtruths that have the given label key
    gts_query = (
        select(
            models.GroundTruthClassification.datum_id.label("datum_id"),
            models.Label.value.label("label_value"),
        )
        .join(models.Datum)
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(
            models.Label,
            and_(
                models.Label.key == label.key,
                models.GroundTruthClassification.label_id == models.Label.id,
            ),
        )
    ).subquery()

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

    # get the prediction scores for the given label (key and value)
    preds_query = (
        select(
            models.PredictedClassification.datum_id.label("datum_id"),
            models.PredictedClassification.score.label("score"),
            models.Label.value.label("label_value"),
        )
        .join(models.Datum)
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(models.Model, models.Model.name == model_name)
        .join(
            models.Label,
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
                models.PredictedClassification.label_id == models.Label.id,
            ),
        )
    ).subquery()

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


def roc_auc(
    db: Session, dataset_name: str, model_name: str, label_key: str
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

    Returns
    -------
    float
        ROC AUC
    """

    labels = [
        label
        for label in crud.get_classification_labels_in_dataset(
            db, dataset_name
        )
        if label.key == label_key
    ]
    if len(labels) == 0:
        raise RuntimeError(
            f"The label key '{label_key}' is not a classification label in the dataset {dataset_name}."
        )

    sum_roc_aucs = 0
    for label in labels:
        sum_roc_aucs += binary_roc_auc(db, dataset_name, model_name, label)

    return sum_roc_aucs / len(labels)


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
    gt_labels = crud.get_classification_labels_in_dataset(db, dataset_name)
    pred_labels = crud.get_classification_prediction_labels(
        db, model_name=model_name, dataset_name=dataset_name
    )
    labels = gt_labels + pred_labels
    unique_label_keys = set([label.key for label in labels])

    confusion_matrices, metrics = [], []
    for label_key in unique_label_keys:
        confusion_matrix = confusion_matrix_at_label_key(
            db, dataset_name, model_name, label_key
        )
        if confusion_matrix is None:
            continue
        confusion_matrices.append(confusion_matrix)

        metrics.append(
            schemas.AccuracyMetric(
                label_key=label_key, value=accuracy_from_cm(confusion_matrix)
            )
        )

        metrics.append(
            schemas.ROCAUCMetric(
                label_key=label_key,
                value=roc_auc(db, dataset_name, model_name, label_key),
            )
        )

        for label in [label for label in labels if label.key == label_key]:
            (
                precision,
                recall,
                f1,
            ) = precision_and_recall_f1_from_confusion_matrix(
                confusion_matrix, label.value
            )

            pydantic_label = schemas.Label(key=label.key, value=label.value)
            metrics.append(
                schemas.PrecisionMetric(label=pydantic_label, value=precision)
            )
            metrics.append(
                schemas.RecallMetric(label=pydantic_label, value=recall)
            )
            metrics.append(schemas.F1Metric(label=pydantic_label, value=f1))

    return confusion_matrices, metrics


def confusion_matrix_at_label_key(
    db: Session,
    dataset_name: str,
    model_name: str,
    label_key: str,
    metadatum_id: int = None,
    metadatum_value: str = None,
    metadatum_name: str = None,
) -> schemas.ConfusionMatrix | None:
    """Returns None in the case that there are not common images in the dataset
    that have both a groundtruth and prediction with label key `label_key`
    """
    # this query get's the max score for each Datum for the given label key
    q1 = (
        select(
            func.max(PredictedClassification.score).label("max_score"),
            PredictedClassification.datum_id,
        )
        .join(models.Label)
        .join(models.Datum)
        .join(models.Dataset)
        .join(models.Model)
    )
    if metadatum_id is None:
        q1 = q1.where(
            and_(
                models.Label.key == label_key,
                models.Dataset.name == dataset_name,
                models.Model.id == PredictedClassification.model_id,
                models.Model.name == model_name,
            )
        )
    else:
        q1 = (
            q1.join(models.DatumMetadatumLink)
            .join(models.Metadatum)
            .where(
                and_(
                    models.Label.key == label_key,
                    models.Dataset.name == dataset_name,
                    models.Model.id == PredictedClassification.model_id,
                    models.Model.name == model_name,
                    models.Metadatum.id == metadatum_id,
                    models.Datum.id == models.DatumMetadatumLink.datum_id,
                )
            )
        )

    q1 = q1.group_by(PredictedClassification.datum_id)
    subquery = q1.alias()

    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    q2 = (
        select(func.min(models.PredictedClassification.id).label("min_id"))
        .join(models.Label)
        .join(
            subquery,
            and_(
                PredictedClassification.score == subquery.c.max_score,
                PredictedClassification.datum_id == subquery.c.datum_id,
            ),
        )
        .group_by(models.PredictedClassification.datum_id)
    )
    min_id_query = q2.alias()

    q3 = (
        select(
            models.Label.value.label("pred_label_value"),
            models.PredictedClassification.datum_id.label("datum_id"),
        )
        .join(models.PredictedClassification)
        .join(
            subquery,
            and_(
                PredictedClassification.score == subquery.c.max_score,
                PredictedClassification.datum_id == subquery.c.datum_id,
            ),
        )
        .join(
            min_id_query,
            PredictedClassification.id == min_id_query.c.min_id,
        )
    )
    hard_preds_query = q3.alias()

    b = Bundle("cols", hard_preds_query.c.pred_label_value, models.Label.value)

    total_query = (
        select(b, func.count())
        .join(
            models.GroundTruthClassification,
            models.GroundTruthClassification.datum_id
            == hard_preds_query.c.datum_id,
        )
        .join(
            models.Label,
            and_(
                models.Label.id == models.GroundTruthClassification.label_id,
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

    if metadatum_id is not None:
        metadatum = schemas.DatumMetadatum(
            name=metadatum_name, value=metadatum_value
        )
    else:
        metadatum = None

    return schemas.ConfusionMatrix(
        label_key=label_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
        metadatum=metadatum,
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

    prec = true_positives / n_preds
    recall = true_positives / n_gts

    f1_denom = prec + recall
    if f1_denom == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / f1_denom
    return prec, recall, f1
