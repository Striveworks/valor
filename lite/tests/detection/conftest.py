import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon
from valor_lite.detection import Bitmask, BoundingBox, Detection, Polygon


@pytest.fixture
def rect1() -> tuple[float, float, float, float]:
    """Box with area = 1500."""
    return (10.0, 60.0, 10.0, 40.0)


@pytest.fixture
def rect2() -> tuple[float, float, float, float]:
    """Box with area = 1100."""
    return (15.0, 70.0, 0.0, 20.0)


@pytest.fixture
def rect3() -> tuple[float, float, float, float]:
    """Box with area = 57,510."""
    return (87.0, 158.0, 10.0, 820.0)


@pytest.fixture
def rect4() -> tuple[float, float, float, float]:
    """Box with area = 90."""
    return (1.0, 10.0, 10.0, 20.0)


@pytest.fixture
def rect5() -> tuple[float, float, float, float]:
    """Box with partial overlap to rect3."""
    return (87, 158, 10, 400)


@pytest.fixture
def rect1_rotated_5_degrees_around_origin() -> list[tuple[float, float]]:
    """Box with area = 1500."""
    return [
        (9.090389553440874, 10.833504408394036),
        (58.90012445802815, 15.191291545776945),
        (56.28545217559841, 45.07713248852931),
        (6.475717271011129, 40.7193453511464),
        (9.090389553440874, 10.833504408394036),
    ]


@pytest.fixture
def rect2_rotated_5_degrees_around_origin() -> list[tuple[float, float]]:
    """Box with area = 1100."""
    return [
        (14.942920471376183, 1.3073361412148725),
        (69.7336288664222, 6.1009019923360714),
        (67.99051401146903, 26.024795954170983),
        (13.19980561642302, 21.231230103049782),
        (14.942920471376183, 1.3073361412148725),
    ]


@pytest.fixture
def rect3_rotated_5_degrees_around_origin() -> list[tuple[float, float]]:
    """Box with area = 57,510."""
    return [
        (85.79738130650527, 17.544496599963715),
        (156.52720487101922, 23.732554335047446),
        (85.9310532454161, 830.6502597893614),
        (15.20122968090216, 824.4622020542777),
        (85.79738130650527, 17.544496599963715),
    ]


@pytest.fixture
def basic_detections_first_class(
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
) -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v1"],
                ),
            ],
            predictions=[],
        ),
    ]


@pytest.fixture
def basic_detections_second_class(
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
) -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect3[0],
                    xmax=rect3[1],
                    ymin=rect3[2],
                    ymax=rect3[3],
                    labels=["v2"],
                ),
            ],
            predictions=[],
        ),
        Detection(
            uid="uid2",
            groundtruths=[],
            predictions=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v2"],
                    scores=[0.98],
                ),
            ],
        ),
    ]


@pytest.fixture
def basic_detections(
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
) -> list[Detection]:
    """Combines the labels from basic_detections_first_class and basic_detections_second_class."""
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
                BoundingBox(
                    xmin=rect3[0],
                    xmax=rect3[1],
                    ymin=rect3[2],
                    ymax=rect3[3],
                    labels=["v2"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v2"],
                    scores=[0.98],
                ),
            ],
        ),
    ]


@pytest.fixture
def basic_rotated_detections_first_class(
    rect1_rotated_5_degrees_around_origin: tuple[float, float, float, float],
    rect2_rotated_5_degrees_around_origin: tuple[float, float, float, float],
) -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                ),
            ],
            predictions=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect2_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                ),
            ],
            predictions=[],
        ),
    ]


@pytest.fixture
def basic_rotated_detections_second_class(
    rect2_rotated_5_degrees_around_origin: tuple[float, float, float, float],
    rect3_rotated_5_degrees_around_origin: tuple[float, float, float, float],
) -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect3_rotated_5_degrees_around_origin
                    ),
                    labels=["v2"],
                ),
            ],
            predictions=[],
        ),
        Detection(
            uid="uid2",
            groundtruths=[],
            predictions=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect2_rotated_5_degrees_around_origin
                    ),
                    labels=["v2"],
                    scores=[0.98],
                ),
            ],
        ),
    ]


@pytest.fixture
def torchmetrics_detections() -> list[Detection]:
    """Creates a model called "test_model" with some predicted
    detections on the dataset "test_dataset". These predictions are taken
    from a torchmetrics unit test (see test_metrics.py)
    """

    # predictions for four images taken from
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L59

    groundtruths = [
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
    predictions = [
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
        Detection(
            uid=str(idx),
            groundtruths=[
                BoundingBox(
                    xmin=box[0],
                    ymin=box[1],
                    xmax=box[2],
                    ymax=box[3],
                    labels=[label_value],
                )
                for box, label_value in zip(gt["boxes"], gt["labels"])
            ],
            predictions=[
                BoundingBox(
                    xmin=box[0],
                    ymin=box[1],
                    xmax=box[2],
                    ymax=box[3],
                    labels=[label_value],
                    scores=[score],
                )
                for box, label_value, score in zip(
                    pd["boxes"], pd["labels"], pd["scores"]
                )
            ],
        )
        for idx, (gt, pd) in enumerate(zip(groundtruths, predictions))
    ]


@pytest.fixture
def false_negatives_single_datum_baseline_detections() -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
                BoundingBox(
                    xmin=100,
                    xmax=110,
                    ymin=100,
                    ymax=200,
                    labels=["value"],
                    scores=[0.7],
                ),
            ],
        )
    ]


@pytest.fixture
def false_negatives_single_datum_detections() -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
                BoundingBox(
                    xmin=100,
                    xmax=110,
                    ymin=100,
                    ymax=200,
                    labels=["value"],
                    scores=[0.9],
                ),
            ],
        )
    ]


@pytest.fixture
def false_negatives_two_datums_one_empty_low_confidence_of_fp_detections() -> (
    list[Detection]
):

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.7],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_datums_one_empty_high_confidence_of_fp_detections() -> (
    list[Detection]
):

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.9],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections() -> (
    list[Detection]
):

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["other value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.7],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections() -> (
    list[Detection]
):

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.8],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["other value"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["value"],
                    scores=[0.9],
                ),
            ],
        ),
    ]


@pytest.fixture
def detections_fp_hallucination_edge_case() -> list[Detection]:
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=0,
                    xmax=5,
                    ymin=0,
                    ymax=5,
                    labels=["v1"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=0,
                    xmax=5,
                    ymin=0,
                    ymax=5,
                    labels=["v1"],
                    scores=[0.8],
                )
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=0,
                    xmax=5,
                    ymin=0,
                    ymax=5,
                    labels=["v1"],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["v1"],
                    scores=[0.8],
                )
            ],
        ),
    ]


@pytest.fixture
def detections_tp_deassignment_edge_case() -> list[Detection]:
    return [
        Detection(
            uid="uid0",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["v1"],
                ),
                BoundingBox(
                    xmin=10,
                    xmax=15,
                    ymin=20,
                    ymax=25,
                    labels=["v1"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=["v1"],
                    scores=[0.78],
                ),
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=12,
                    ymax=22,
                    labels=["v1"],
                    scores=[0.96],
                ),
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=12,
                    ymax=22,
                    labels=["v1"],
                    scores=[0.96],
                ),
                BoundingBox(
                    xmin=101,
                    xmax=102,
                    ymin=101,
                    ymax=102,
                    labels=["v1"],
                    scores=[0.87],
                ),
            ],
        ),
    ]


@pytest.fixture
def detection_ranked_pair_ordering() -> Detection:

    gts = {
        "boxes": [
            (2, 10, 2, 10),
            (2, 10, 2, 10),
            (2, 10, 2, 10),
        ],
        "label_values": ["label1", "label2", "label3"],
    }

    # labels 1 and 2 have IOU==1, labels 3 and 4 have IOU==0
    preds = {
        "boxes": [
            (2, 10, 2, 10),
            (2, 10, 2, 10),
            (0, 1, 0, 1),
            (0, 1, 0, 1),
        ],
        "label_values": ["label1", "label2", "label3", "label4"],
        "scores": [
            0.3,
            0.93,
            0.92,
            0.94,
        ],
    }

    groundtruths = [
        BoundingBox(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            labels=[label_value],
        )
        for (xmin, xmax, ymin, ymax), label_value in zip(
            gts["boxes"], gts["label_values"]
        )
    ]

    predictions = [
        BoundingBox(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            labels=[label_value],
            scores=[score],
        )
        for (xmin, xmax, ymin, ymax), label_value, score in zip(
            preds["boxes"], preds["label_values"], preds["scores"]
        )
    ]

    return Detection(
        uid="uid1", groundtruths=groundtruths, predictions=predictions
    )


@pytest.fixture
def detection_ranked_pair_ordering_with_bitmasks() -> Detection:

    bitmask1 = np.zeros((100, 50), dtype=np.bool_)
    bitmask1[79, 31] = True

    bitmask2 = np.zeros((100, 50), dtype=np.bool_)
    bitmask2[80:, 32:] = True

    gts = {
        "bitmasks": [
            bitmask1,
            bitmask1,
            bitmask1,
        ],
        "label_values": ["label1", "label2", "label3"],
    }

    # labels 1 and 2 have IOU==1, labels 3 and 4 have IOU==0
    preds = {
        "bitmasks": [bitmask1, bitmask1, bitmask2, bitmask2],
        "label_values": ["label1", "label2", "label3", "label4"],
        "scores": [
            0.3,
            0.93,
            0.92,
            0.94,
        ],
    }

    groundtruths = [
        Bitmask(
            mask=mask,
            labels=[label_value],
        )
        for mask, label_value in zip(gts["bitmasks"], gts["label_values"])
    ]

    predictions = [
        Bitmask(
            mask=mask,
            labels=[label_value],
            scores=[score],
        )
        for mask, label_value, score in zip(
            preds["bitmasks"], preds["label_values"], preds["scores"]
        )
    ]

    return Detection(
        uid="uid1", groundtruths=groundtruths, predictions=predictions
    )


@pytest.fixture
def detection_ranked_pair_ordering_with_polygons(
    rect1_rotated_5_degrees_around_origin: list[tuple[float, float]],
    rect3_rotated_5_degrees_around_origin: list[tuple[float, float]],
) -> Detection:
    gts = {
        "polygons": [
            rect1_rotated_5_degrees_around_origin,
            rect1_rotated_5_degrees_around_origin,
            rect1_rotated_5_degrees_around_origin,
        ],
        "label_values": ["label1", "label2", "label3"],
    }

    # labels 1 and 2 have IOU==1, labels 3 and 4 have IOU==0
    preds = {
        "polygons": [
            rect1_rotated_5_degrees_around_origin,
            rect1_rotated_5_degrees_around_origin,
            rect3_rotated_5_degrees_around_origin,
            rect3_rotated_5_degrees_around_origin,
        ],
        "label_values": ["label1", "label2", "label3", "label4"],
        "scores": [
            0.3,
            0.93,
            0.92,
            0.94,
        ],
    }

    groundtruths = [
        Polygon(
            shape=ShapelyPolygon(polygon),
            labels=[label_value],
        )
        for polygon, label_value in zip(gts["polygons"], gts["label_values"])
    ]

    predictions = [
        Polygon(
            shape=ShapelyPolygon(polygon),
            labels=[label_value],
            scores=[score],
        )
        for polygon, label_value, score in zip(
            preds["polygons"], preds["label_values"], preds["scores"]
        )
    ]

    return Detection(
        uid="uid1", groundtruths=groundtruths, predictions=predictions
    )


@pytest.fixture
def detections_no_groundtruths() -> list[Detection]:
    return [
        Detection(
            uid="uid",
            groundtruths=[],
            predictions=[
                BoundingBox(
                    xmin=0,
                    xmax=10,
                    ymin=0,
                    ymax=10,
                    labels=["v1"],
                    scores=[1.0],
                ),
            ],
        ),
        Detection(
            uid="uid",
            groundtruths=[],
            predictions=[
                BoundingBox(
                    xmin=0,
                    xmax=10,
                    ymin=0,
                    ymax=10,
                    labels=["v1"],
                    scores=[1.0],
                ),
            ],
        ),
    ]


@pytest.fixture
def detections_no_predictions() -> list[Detection]:
    return [
        Detection(
            uid="uid",
            groundtruths=[
                BoundingBox(xmin=0, xmax=10, ymin=0, ymax=10, labels=["v1"]),
            ],
            predictions=[],
        ),
        Detection(
            uid="uid",
            groundtruths=[
                BoundingBox(xmin=0, xmax=10, ymin=0, ymax=10, labels=["v1"]),
            ],
            predictions=[],
        ),
    ]


@pytest.fixture
def detections_for_detailed_counting(
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
    rect4: tuple[float, float, float, float],
    rect5: tuple[float, float, float, float],
) -> list[Detection]:

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["missed_detection"],
                ),
                BoundingBox(
                    xmin=rect3[0],
                    xmax=rect3[1],
                    ymin=rect3[2],
                    ymax=rect3[3],
                    labels=["v2"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                    scores=[0.5],
                ),
                BoundingBox(
                    xmin=rect5[0],
                    xmax=rect5[1],
                    ymin=rect5[2],
                    ymax=rect5[3],
                    labels=["not_v2"],
                    scores=[0.3],
                ),
                BoundingBox(
                    xmin=rect4[0],
                    xmax=rect4[1],
                    ymin=rect4[2],
                    ymax=rect4[3],
                    labels=["hallucination"],
                    scores=[0.1],
                ),
            ],
        ),
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["low_iou"],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["low_iou"],
                    scores=[0.5],
                ),
            ],
        ),
    ]
