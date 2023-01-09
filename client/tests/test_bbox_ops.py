import pytest
from velour.bbox_ops import _match_array, iou_matrix
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
    return [
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


@pytest.fixture
def predictions():
    return [
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


def test__match_array(groundtruths, predictions):
    # note that we're ignoring the class labels here
    predictions = sorted(predictions, key=lambda g: g.score, reverse=True)

    ious = iou_matrix(groundtruths=groundtruths, predictions=predictions)

    assert _match_array(ious, 1.0) == [None, None, None, None]

    assert _match_array(ious, 0.75) == [2, None, 1, None]

    assert _match_array(ious, 0.7) == [2, None, 1, 0]

    # check that match to groundtruth 0 switches
    assert _match_array(ious, 0.1) == [2, 0, 1, None]

    assert _match_array(ious, 0.0) == [2, 0, 1, None]
