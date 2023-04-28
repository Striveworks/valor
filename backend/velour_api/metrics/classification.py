import numpy as np


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
