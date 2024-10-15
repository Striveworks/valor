from valor_api import schemas
from valor_api.backend.metrics.classification import (
    _compute_accuracy_from_cm,
    _compute_precision_and_recall_f1_from_confusion_matrix,
)


def test__compute_precision_and_recall_f1_from_confusion_matrix(
    cm: schemas.ConfusionMatrix,
):
    """c.f. with

    ```
    from sklearn.metrics import classification_report

    y_true = [0, 0, 0, 1, 2, 2]
    y_pred = [0, 1, 2, 1, 1, 1]

    print(classification_report(y_true, y_pred))
    ```
    """
    (
        prec,
        recall,
        f1,
    ) = _compute_precision_and_recall_f1_from_confusion_matrix(cm, "class0")
    assert prec == 1.0
    assert recall == 1 / 3
    assert f1 == 0.5

    (
        prec,
        recall,
        f1,
    ) = _compute_precision_and_recall_f1_from_confusion_matrix(cm, "class1")
    assert prec == 0.25
    assert recall == 1.0
    assert f1 == 0.4

    (
        prec,
        recall,
        f1,
    ) = _compute_precision_and_recall_f1_from_confusion_matrix(cm, "class2")
    assert prec == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test__compute_accuracy_from_cm(cm: schemas.ConfusionMatrix):
    assert _compute_accuracy_from_cm(cm) == 1 / 3
