from velour_api import schemas
from velour_api.metrics.classification import (
    accuracy_from_cm,
    precision_and_recall_f1_from_confusion_matrix,
)
from velour_api.metrics.detection import RankedPair, _ap


def truncate_float(x: float) -> str:
    return f"{int(x)}.{int((x - int(x)) * 100)}"


def test__ap():

    pairs = {
        "0": [
            RankedPair(1, 1, score=0.8, iou=0.6),
            RankedPair(2, 2, score=0.6, iou=0.8),
            RankedPair(3, 3, score=0.4, iou=1.0),
        ],
        "1": [
            RankedPair(0, 0, score=0.0, iou=1.0),
            RankedPair(2, 2, score=0.0, iou=1.0),
        ],
        "2": [
            RankedPair(0, 0, score=1.0, iou=1.0),
        ],
    }

    labels = {
        "0": schemas.Label(key="name", value="car"),
        "1": schemas.Label(key="name", value="dog"),
        "2": schemas.Label(key="name", value="person"),
    }

    number_of_ground_truths = {
        "0": 3,
        "1": 2,
        "2": 4,
    }

    iou_thresholds = [0.5, 0.75, 0.9]

    # Calculated by hand
    reference_metrics = [
        schemas.APMetric(
            iou=0.5, value=1.0, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.5, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.5,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
        schemas.APMetric(
            iou=0.75, value=0.44, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.75, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.75,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
        schemas.APMetric(
            iou=0.9, value=0.11, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.9, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.9,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
    ]

    ap_metrics = _ap(
        sorted_ranked_pairs=pairs,
        number_of_ground_truths=number_of_ground_truths,
        labels=labels,
        iou_thresholds=iou_thresholds,
    )

    assert len(reference_metrics) == len(ap_metrics)
    for pd, gt in zip(ap_metrics, reference_metrics):
        assert pd.iou == gt.iou
        assert truncate_float(pd.value) == truncate_float(gt.value)
        assert pd.label == gt.label


def test_precision_and_recall_f1_from_confusion_matrix(
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
    ) = precision_and_recall_f1_from_confusion_matrix(cm, "class0")
    assert prec == 1.0
    assert recall == 1 / 3
    assert f1 == 0.5

    (
        prec,
        recall,
        f1,
    ) = precision_and_recall_f1_from_confusion_matrix(cm, "class1")
    assert prec == 0.25
    assert recall == 1.0
    assert f1 == 0.4

    (
        prec,
        recall,
        f1,
    ) = precision_and_recall_f1_from_confusion_matrix(cm, "class2")
    assert prec == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_accuracy_from_cm(cm: schemas.ConfusionMatrix):
    assert accuracy_from_cm(cm) == 1 / 3
