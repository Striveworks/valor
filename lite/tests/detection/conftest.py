import pytest
from valor_lite.detection import BoundingBox, Detection


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
def basic_detections(
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
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
                    labels=[("k1", "v1")],
                ),
                BoundingBox(
                    xmin=rect3[0],
                    xmax=rect3[1],
                    ymin=rect3[2],
                    ymax=rect3[3],
                    labels=[("k2", "v2")],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=[("k1", "v1")],
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
                    labels=[("k1", "v1")],
                ),
            ],
            predictions=[
                BoundingBox(
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=[("k2", "v2")],
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
                    labels=[("class", label_value)],
                )
                for box, label_value in zip(gt["boxes"], gt["labels"])
            ],
            predictions=[
                BoundingBox(
                    xmin=box[0],
                    ymin=box[1],
                    xmax=box[2],
                    ymax=box[3],
                    labels=[("class", label_value)],
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
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                    scores=[0.8],
                ),
                BoundingBox(
                    xmin=100,
                    xmax=110,
                    ymin=100,
                    ymax=200,
                    labels=[("key", "value")],
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
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                    scores=[0.8],
                ),
                BoundingBox(
                    xmin=100,
                    xmax=110,
                    ymin=100,
                    ymax=200,
                    labels=[("key", "value")],
                    scores=[0.9],
                ),
            ],
        )
    ]


@pytest.fixture
def false_negatives_two_datums_one_empty_low_confidence_of_fp_detections() -> list[
    Detection
]:

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
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
                    labels=[("key", "value")],
                    scores=[0.7],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_datums_one_empty_high_confidence_of_fp_detections() -> list[
    Detection
]:

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
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
                    labels=[("key", "value")],
                    scores=[0.9],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections() -> list[
    Detection
]:

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
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
                    labels=[("key", "other value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                    scores=[0.7],
                ),
            ],
        ),
    ]


@pytest.fixture
def false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections() -> list[
    Detection
]:

    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
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
                    labels=[("key", "other value")],
                )
            ],
            predictions=[
                BoundingBox(
                    xmin=10,
                    xmax=20,
                    ymin=10,
                    ymax=20,
                    labels=[("key", "value")],
                    scores=[0.9],
                ),
            ],
        ),
    ]
