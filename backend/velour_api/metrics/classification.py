import numpy as np
from sqlalchemy import Float, Integer
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, func, select

from velour_api import crud, models, schemas
from velour_api.models import (
    GroundTruthImageClassification,
    PredictedImageClassification,
)


def precision_and_recall_f1_at_class_index_from_confusion_matrix(
    cm: np.ndarray, class_index: int
) -> float:
    """Computes the precision, recall, and f1 score at a class index"""
    true_positives = cm[class_index, class_index]
    # number of times the class was predicted
    n_preds = cm[:, class_index].sum()
    n_gts = cm[class_index, :].sum()

    prec = true_positives / n_preds
    recall = true_positives / n_gts

    f1_denom = prec + recall
    if f1_denom == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / f1_denom
    return prec, recall, f1


def binary_roc_auc(
    db: Session, dataset_name: str, label: schemas.Label
) -> float:
    # query to get the image_ids and label values of groundtruths that have the given label key
    gts_query = (
        select(
            models.GroundTruthImageClassification.image_id.label("image_id"),
            models.Label.value.label("label_value"),
        )
        .join(models.Image)
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(
            models.Label,
            and_(
                models.Label.key == label.key,
                models.GroundTruthImageClassification.label_id
                == models.Label.id,
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

    # get the prediction scores for the given label (key and value)
    preds_query = (
        select(
            models.PredictedImageClassification.image_id.label("image_id"),
            models.PredictedImageClassification.score.label("score"),
            models.Label.value.label("label_value"),
        )
        .join(models.Image)
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(
            models.Label,
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
                models.PredictedImageClassification.label_id
                == models.Label.id,
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
        ).join(preds_query, gts_query.c.image_id == preds_query.c.image_id)
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

    return db.scalar(func.sum(trap_areas.c.trap_area))


def roc_auc(db: Session, dataset_name: str, label_key: str) -> float:
    """Comptues the area under the ROC curve. Note that for the multi-class setting
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
        sum_roc_aucs += binary_roc_auc(db, dataset_name, label)

    return sum_roc_aucs / len(labels)


def compute_clf_metrics(
    db: Session,
    predictions: list[list[PredictedImageClassification]],
    groundtruths: list[list[GroundTruthImageClassification]],
):
    # Return: accuracies (one for each label key)
    # confusion matrices (one for each label key)
    # roc_auc (one for each label key)
    pass


def confusion_matrix_at_label_key(
    db: Session, dataset_name: str, label_key: str
) -> list[dict[str, str | int]]:
    subquery = (
        select(
            func.max(PredictedImageClassification.score).label("max_score"),
            func.min(PredictedImageClassification.id).label(
                "min_id"
            ),  # this is for the corner case where the maximum score occurs twice
        )
        .join(models.Label)
        .join(models.Image)
        .join(models.Dataset)
        .where(
            and_(
                models.Label.key == label_key,
                models.Dataset.name == dataset_name,
            )
        )
        .group_by(PredictedImageClassification.image_id)
        .alias()
    )

    hard_preds_query = (
        select(
            models.Label.value.label("pred_label_value"),
            models.PredictedImageClassification.image_id.label("image_id"),
        )
        .join(models.PredictedImageClassification)
        .join(
            subquery,
            and_(
                PredictedImageClassification.score == subquery.c.max_score,
                PredictedImageClassification.id == subquery.c.min_id,
            ),
        )
        .alias()
    )

    b = Bundle("cols", hard_preds_query.c.pred_label_value, models.Label.value)

    total_query = (
        select(b, func.count())
        .join(
            models.GroundTruthImageClassification,
            models.GroundTruthImageClassification.image_id
            == hard_preds_query.c.image_id,
        )
        .join(
            models.Label,
            and_(
                models.Label.id
                == models.GroundTruthImageClassification.label_id,
                models.Label.key == label_key,
            ),
        )
        .group_by(b)
    )

    res = db.execute(total_query).all()

    return schemas.ConfusionMatrix(
        label_key=label_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
    )


def accuracy_from_cm(cm: schemas.ConfusionMatrix) -> float:
    num, denom = 0, 0
    for entry in cm.entries:
        denom += entry.count
        if entry.prediction == entry.groundtruth:
            num += entry.count

    return num / denom
