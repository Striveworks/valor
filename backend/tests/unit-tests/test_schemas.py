import pytest

from velour_api.schemas import GroundTruthDetectionCreate, Image, Label


def test_ground_truth_detection_validation_pos():
    boundary = [[1, 1], [2, 2], [0, 4]]
    det = GroundTruthDetectionCreate(
        boundary=boundary,
        class_label="class",
        labels=[Label(key="class", value="a")],
        image=Image(uri=""),
    )
    assert det.boundary == [tuple(pt) for pt in boundary]


def test_ground_truth_detection_validation_neg():
    boundary = [[1, 1], [2, 2]]
    with pytest.raises(ValueError) as exc_info:
        GroundTruthDetectionCreate(
            boundary=boundary,
            class_label="class",
            labels=[Label(key="class", value="a")],
            image=Image(uri=""),
        )
    assert "must be composed of at least three points" in str(exc_info)
