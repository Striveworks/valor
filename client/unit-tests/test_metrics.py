from typing import List

import numpy as np
import pytest

from valor.coretypes import Annotation, Datum, GroundTruth, Label, Prediction
from valor.metrics.classification import (
    combine_tps_fps_thresholds,
    get_tps_fps_thresholds,
)
from valor.metrics.detection import (
    compute_ap_metrics,
    evaluate_detection,
    get_intermediate_metric_data,
)
from valor.schemas import Box


def box_list_to_box(bl: List[float]) -> Box:
    return Box.from_extrema(xmin=bl[0], ymin=bl[1], xmax=bl[2], ymax=bl[3])


@pytest.fixture
def groundtruths() -> List[GroundTruth]:
    gts_per_img = [
        {"boxes": [[214.1500, 41.2900, 562.4100, 285.0700]], "labels": ["4"]},
        {
            "boxes": [
                [13.00, 22.75, 548.98, 632.42],
                [1.66, 3.32, 270.26, 275.23],
            ],
            "labels": ["2", "2"],
        },
        {
            "boxes": [
                [61.87, 276.25, 358.29, 379.43],
                [2.75, 3.66, 162.15, 316.06],
                [295.55, 93.96, 313.97, 152.79],
                [326.94, 97.05, 340.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [462.08, 105.09, 493.74, 146.99],
                [277.11, 103.84, 292.44, 150.72],
            ],
            "labels": ["4", "1", "0", "0", "0", "0", "0"],
        },
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [50.17, 45.34, 71.28, 79.83],
                [81.28, 47.04, 98.66, 78.50],
                [63.96, 46.17, 84.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [56.39, 21.65, 75.66, 45.54],
                [73.14, 1.10, 98.96, 28.33],
                [62.34, 55.23, 78.14, 79.57],
                [44.17, 45.78, 63.99, 78.48],
                [58.18, 44.80, 66.42, 56.25],
            ],
            "labels": [
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
            ],
        },
    ]
    return [
        GroundTruth(
            datum=Datum(uid=str(i)),
            annotations=[
                Annotation(
                    bounding_box=box_list_to_box(box),
                    labels=[Label(key="class", value=class_label)],
                )
                for box, class_label in zip(gts["boxes"], gts["labels"])
            ],
        )
        for i, gts in enumerate(gts_per_img)
    ]


@pytest.fixture
def predictions() -> List[Prediction]:
    # predictions for four images taken from
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L59
    preds_per_img = [
        {
            "boxes": [[258.15, 41.29, 606.41, 285.07]],
            "scores": [0.236],
            "labels": ["4"],
        },
        {
            "boxes": [
                [61.00, 22.75, 565.00, 632.42],
                [12.66, 3.32, 281.26, 275.23],
            ],
            "scores": [0.318, 0.726],
            "labels": ["3", "2"],
        },
        {
            "boxes": [
                [87.87, 276.25, 384.29, 379.43],
                [0.00, 3.66, 142.15, 316.06],
                [296.55, 93.96, 314.97, 152.79],
                [328.94, 97.05, 342.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [464.08, 105.09, 495.74, 146.99],
                [276.11, 103.84, 291.44, 150.72],
            ],
            "scores": [0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953],
            "labels": ["4", "1", "0", "0", "0", "0", "0"],
        },
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [45.17, 45.34, 66.28, 79.83],
                [82.28, 47.04, 99.66, 78.50],
                [59.96, 46.17, 80.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [71.14, 1.10, 96.96, 28.33],
                [61.34, 55.23, 77.14, 79.57],
                [41.17, 45.78, 60.99, 78.48],
                [56.18, 44.80, 64.42, 56.25],
            ],
            "scores": [
                0.532,
                0.204,
                0.782,
                0.202,
                0.883,
                0.271,
                0.561,
                0.204,
                0.349,
            ],
            "labels": ["49", "49", "49", "49", "49", "49", "49", "49", "49"],
        },
    ]

    return [
        Prediction(
            datum=Datum(uid=str(i)),
            annotations=[
                Annotation(
                    bounding_box=box_list_to_box(box),
                    labels=[
                        Label(key="class", value=class_label, score=score)
                    ],
                )
                for box, class_label, score in zip(
                    preds["boxes"], preds["labels"], preds["scores"]
                )
            ],
        )
        for i, preds in enumerate(preds_per_img)
    ]


@pytest.fixture
def expected_ap_metrics() -> dict:
    return {
        "AP": {
            ("class", "2"): {
                "IoU=0.5": 0.505,
                "IoU=0.75": 0.505,
                "IoU=0.5:0.95": 0.454,
            },
            ("class", "49"): {
                "IoU=0.5": 0.79,
                "IoU=0.75": 0.576,
                "IoU=0.5:0.95": 0.555,
            },
            ("class", "3"): {
                "IoU=0.5": -1.0,
                "IoU=0.75": -1.0,
                "IoU=0.5:0.95": -1.0,
            },
            ("class", "0"): {
                "IoU=0.5": 1.0,
                "IoU=0.75": 0.723,
                "IoU=0.5:0.95": 0.725,
            },
            ("class", "1"): {
                "IoU=0.5": 1.0,
                "IoU=0.75": 1.0,
                "IoU=0.5:0.95": 0.8,
            },
            ("class", "4"): {
                "IoU=0.5": 1.0,
                "IoU=0.75": 1.0,
                "IoU=0.5:0.95": 0.65,
            },
        },
        "mAP": {"IoU=0.5": 0.859, "IoU=0.75": 0.761, "IoU=0.5:0.95": 0.637},
    }


def round_dict_(d: dict, prec: int) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            round_dict_(v, prec)


def test_compute_ap_metrics(
    groundtruths: List[GroundTruth],
    predictions: List[Prediction],
    expected_ap_metrics: dict,
):
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    metrics = compute_ap_metrics(
        predictions=predictions,
        groundtruths=groundtruths,
        iou_thresholds=iou_thresholds,
    )

    for iou_thres in [i for i in iou_thresholds if i not in [0.5, 0.75]]:
        k = f"IoU={iou_thres}"
        metrics["mAP"].pop(k)
        for class_label in metrics["AP"].keys():
            metrics["AP"][class_label].pop(k)

    round_dict_(metrics, 3)

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    assert metrics == expected_ap_metrics


def test_compute_ap_metrics_in_pieces(
    groundtruths: List[GroundTruth],
    predictions: List[Prediction],
    expected_ap_metrics: dict,
):
    """Check that we can compute the AP metrics by chunking out the images and then
    aggregating the intermediate data
    """
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    gts1 = groundtruths[:2]
    preds1 = predictions[:2]
    gts2 = groundtruths[2:]
    preds2 = predictions[2:]

    intermediate1 = get_intermediate_metric_data(preds1, gts1, iou_thresholds)
    intermediate2 = get_intermediate_metric_data(preds2, gts2, iou_thresholds)

    metrics = evaluate_detection(intermediate1, intermediate2)

    round_dict_(metrics, 3)

    for iou_thres in [i for i in iou_thresholds if i not in [0.5, 0.75]]:
        k = f"IoU={iou_thres}"
        for class_label in metrics["AP"].keys():
            metrics["AP"][class_label].pop(k)
    assert metrics["AP"] == expected_ap_metrics["AP"]


def test_get_tps_fps_thresholds():
    y_true = np.array([True, False, True, True, False])
    y_score = np.array([0.8, 0.9, 0.7, 0.5, 0.6])

    tps, fps, thresholds = get_tps_fps_thresholds(y_true, y_score)

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))


def test_combine_tps_fps_thresholds():
    y_true1 = np.array([True, True])
    y_score1 = np.array([0.8, 0.5])

    y_true2 = np.array([False, False, True])
    y_score2 = np.array([0.9, 0.6, 0.7])

    tps1, fps1, thresholds1 = get_tps_fps_thresholds(y_true1, y_score1)
    tps2, fps2, thresholds2 = get_tps_fps_thresholds(y_true2, y_score2)

    tps, fps, thresholds = combine_tps_fps_thresholds(
        tps1, fps1, thresholds1, tps2, fps2, thresholds2
    )

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))


def test_combine_tps_fps_thresholds_dup_threshold():
    y_true1 = np.array([True, True])
    y_score1 = np.array([0.8, 0.5])

    y_true2 = np.array([False, False, True])
    y_score2 = np.array([0.9, 0.5, 0.7])

    tps1, fps1, thresholds1 = get_tps_fps_thresholds(y_true1, y_score1)
    tps2, fps2, thresholds2 = get_tps_fps_thresholds(y_true2, y_score2)

    tps, fps, thresholds = combine_tps_fps_thresholds(
        tps1, fps1, thresholds1, tps2, fps2, thresholds2
    )

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2]))
