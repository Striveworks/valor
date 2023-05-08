from velour_api import schemas
from velour_api.metrics.classification import (
    accuracy_from_cm,
    precision_and_recall_f1_at_class_index_from_confusion_matrix,
)
from velour_api.metrics.detection import _match_array


def test__match_array():
    ious = [
        [0.0, 0.004, 0.78],
        [0.65, 0.0, 0.0],
        [0.0, 0.8, 0.051],
        [0.71, 0.0, 0.0],
    ]

    assert _match_array(ious, 1.0) == [None, None, None, None]

    assert _match_array(ious, 0.75) == [2, None, 1, None]

    assert _match_array(ious, 0.7) == [2, None, 1, 0]

    # check that match to groundtruth 0 switches
    assert _match_array(ious, 0.1) == [2, 0, 1, None]

    assert _match_array(ious, 0.0) == [2, 0, 1, None]


def test_precision_and_recall_f1_at_class_index_from_confusion_matrix(
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
    ) = precision_and_recall_f1_at_class_index_from_confusion_matrix(
        cm, "class0"
    )
    assert prec == 1.0
    assert recall == 1 / 3
    assert f1 == 0.5

    (
        prec,
        recall,
        f1,
    ) = precision_and_recall_f1_at_class_index_from_confusion_matrix(
        cm, "class1"
    )
    assert prec == 0.25
    assert recall == 1.0
    assert f1 == 0.4

    (
        prec,
        recall,
        f1,
    ) = precision_and_recall_f1_at_class_index_from_confusion_matrix(
        cm, "class2"
    )
    assert prec == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_accuracy_from_cm(cm: schemas.ConfusionMatrix):
    assert accuracy_from_cm(cm) == 1 / 3
