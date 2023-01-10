from typing import List

import pytest

from velour.bbox_ops import _match_array, compute_ap_metrics, iou_matrix
from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    Point,
    PredictedDetection,
)


def bounding_box(xmin, ymin, xmax, ymax) -> BoundingPolygon:
    return BoundingPolygon(
        [
            Point(x=xmin, y=ymin),
            Point(x=xmin, y=ymax),
            Point(x=xmax, y=ymax),
            Point(x=xmax, y=ymin),
        ]
    )


@pytest.fixture
def groundtruths():
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
        [
            GroundTruthDetection(
                boundary=bounding_box(*box),
                class_label=class_label,
            )
            for box, class_label in zip(gts["boxes"], gts["labels"])
        ]
        for gts in gts_per_img
    ]


@pytest.fixture
def predictions() -> List[List[PredictedDetection]]:
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
        [
            PredictedDetection(
                boundary=bounding_box(*box),
                class_label=class_label,
                score=score,
            )
            for box, class_label, score in zip(
                preds["boxes"], preds["labels"], preds["scores"]
            )
        ]
        for preds in preds_per_img
    ]


def test__match_array():
    gts = [
        GroundTruthDetection(
            boundary=bounding_box(185, 84, 231, 150), class_label="class 2"
        ),
        GroundTruthDetection(
            boundary=bounding_box(463, 303, 497, 315), class_label="class3"
        ),
        GroundTruthDetection(
            boundary=bounding_box(433, 260, 470, 314), class_label="class 1"
        ),
    ]

    preds = [
        PredictedDetection(
            boundary=bounding_box(433, 259, 464, 311),
            class_label="class 1",
            score=0.9,
        ),
        PredictedDetection(
            boundary=bounding_box(201, 84, 231, 150),
            class_label="class 2",
            score=0.8,
        ),
        PredictedDetection(
            boundary=bounding_box(460, 302, 495, 315),
            class_label="class 3",
            score=0.55,
        ),
        PredictedDetection(
            boundary=bounding_box(184, 85, 219, 150),
            class_label="class 2",
            score=0.4,
        ),
    ]

    # note that we're ignoring the class labels here
    preds = sorted(preds, key=lambda g: g.score, reverse=True)

    ious = iou_matrix(groundtruths=gts, predictions=preds)

    assert _match_array(ious, 1.0) == [None, None, None, None]

    assert _match_array(ious, 0.75) == [2, None, 1, None]

    assert _match_array(ious, 0.7) == [2, None, 1, 0]

    # check that match to groundtruth 0 switches
    assert _match_array(ious, 0.1) == [2, 0, 1, None]

    assert _match_array(ious, 0.0) == [2, 0, 1, None]


def round_dict_(d: dict, prec: int) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            round_dict_(v, prec)


def test_compute_ap_metrics(groundtruths, predictions):
    iou_thresholds = [0.25, 0.5, 0.75]
    metrics = compute_ap_metrics(
        predictions=predictions,
        groundtruths=groundtruths,
        iou_thresholds=iou_thresholds,
    )

    round_dict_(metrics, 3)

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    assert metrics == {
        "AP": {
            "2": {0.25: 0.505, 0.5: 0.505, 0.75: 0.505},
            "49": {0.25: 0.901, 0.5: 0.79, 0.75: 0.576},
            "3": {0.25: -1.0, 0.5: -1.0, 0.75: -1.0},
            "0": {0.25: 1.0, 0.5: 1.0, 0.75: 0.723},
            "1": {0.25: 1.0, 0.5: 1.0, 0.75: 1.0},
            "4": {0.25: 1.0, 0.5: 1.0, 0.75: 1.0},
        },
        "mAP": {0.25: 0.881, 0.5: 0.859, 0.75: 0.761},
    }
