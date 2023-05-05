import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, func, select

from velour_api import models
from velour_api.models import (
    GroundTruthImageClassification,
    PredictedImageClassification,
)


def confusion_matrix(groundtruths: list[str], preds: list[str]) -> np.ndarray:
    """Computes the confusion matrix. The classes are ordered alphabetically and
    for the return array, the first axis corresponds to the groundtruths
    and the second the predictions.
    """
    if len(groundtruths) != len(preds):
        raise RuntimeError(
            "`groundtruths` and `preds` should have the same length."
        )
    unique_labels = sorted(list(set(groundtruths)))
    n_classes = len(unique_labels)
    classes_to_idx = {c: i for i, c in enumerate(unique_labels)}

    ret = np.zeros((n_classes, n_classes), dtype=int)

    for gt, pred in zip(groundtruths, preds):
        gt_idx = classes_to_idx[gt]
        pred_idx = classes_to_idx[pred]

        ret[gt_idx, pred_idx] += 1

    return ret


def accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
    """Computes the accuracy of a confusion matrix"""
    return cm.trace() + cm.sum()


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


def _binary_roc_auc(groundtruths: list[bool], preds: list[float]) -> float:
    # sort the arrays by decreasing order of the scores
    sorted_idxs = np.argsort(preds)[::-1]

    groundtruths = np.array(groundtruths)[sorted_idxs]
    preds = np.array(preds)[sorted_idxs]

    n_gts = groundtruths.sum()
    tps = np.cumsum(groundtruths)
    tprs = tps / n_gts

    fps = np.cumsum(~groundtruths)
    fprs = fps / (len(groundtruths) - n_gts)

    return np.trapz(x=fprs, y=tprs)


def roc_auc(groundtruths: list[str], preds: list[dict[str, float]]) -> float:
    """Comptues the area under the ROC curve. Note that for the multi-class setting
    this does one-vs-rest AUC for each class and then averages those scores.

    Parameters
    ----------
    groundtruths
        list of groundtruth labels
    preds
        list of (soft) predictions. each element should be a dictionary with
        keys equal to the set of unique labels present in groundtruths and values
        the prediction score for that class

    Returns
    -------
    float
        ROC AUC
    """
    if len(groundtruths) != len(preds):
        raise RuntimeError(
            "`groundtruths` and `preds` should have the same length."
        )

    for pred in preds:
        if abs(sum(pred.values()) - 1) >= 1e-5:
            raise ValueError("Sum of predictions should be 1.0")

    unique_classes = set(groundtruths)
    sum_roc_aucs = 0
    for c in unique_classes:
        gts = [gt == c for gt in groundtruths]
        ps = [pred[c] for pred in preds]
        sum_roc_aucs += _binary_roc_auc(groundtruths=gts, preds=ps)

    return sum_roc_aucs / len(unique_classes)


def compute_clf_metrics(
    db: Session,
    predictions: list[list[PredictedImageClassification]],
    groundtruths: list[list[GroundTruthImageClassification]],
):
    # Return: accuracies (one for each label key)
    # confusion matrices (one for each label key)
    # roc_auc (one for each label key)
    pass


def get_hard_preds_from_label_key(
    db: Session, dataset_name: str, label_key: str
):
    subquery = (
        select(
            func.max(PredictedImageClassification.score).label("max_score"),
            func.min(PredictedImageClassification.id).label(
                "min_id"
            ),  # this is for the corner case where the maximum score occurs twice
            PredictedImageClassification.image_id.label("image_id"),
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

    query = select(PredictedImageClassification).join(
        subquery,
        and_(
            PredictedImageClassification.image_id == subquery.c.image_id,
            PredictedImageClassification.score == subquery.c.max_score,
            PredictedImageClassification.id == subquery.c.min_id,
        ),
    )

    return db.scalars(query).all()
