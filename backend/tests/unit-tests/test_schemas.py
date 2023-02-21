import os
from tempfile import TemporaryDirectory

import PIL.Image
import pytest

from velour_api.schemas import (
    DetectionBase,
    Image,
    Label,
    PredictedSegmentation,
)


def test_ground_truth_detection_validation_pos():
    boundary = [[1, 1], [2, 2], [0, 4]]
    det = DetectionBase(
        boundary=boundary,
        labels=[Label(key="class", value="a")],
        image=Image(uri=""),
    )
    assert det.boundary == [tuple(pt) for pt in boundary]


def test_ground_truth_detection_validation_neg():
    boundary = [[1, 1], [2, 2]]
    with pytest.raises(ValueError) as exc_info:
        DetectionBase(
            boundary=boundary,
            labels=[Label(key="class", value="a")],
            image=Image(uri=""),
        )
    assert "must be composed of at least three points" in str(exc_info)


def test_predicted_segmentation_validation_pos():
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode="1", size=(20, 20))
        img_path = os.path.join(tempdir, "img.png")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

    pred_seg = PredictedSegmentation(
        shape=img_bytes, image=Image(uri="uri"), scored_labels=[]
    )
    assert pred_seg.shape == img_bytes


def test_predicted_segmentation_validation_mode_neg():
    """Check we get an error if the mode is not binary"""
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode="RGB", size=(20, 20))
        img_path = os.path.join(tempdir, "img.png")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

    with pytest.raises(ValueError) as exc_info:
        PredictedSegmentation(
            shape=img_bytes, image=Image(uri="uri"), scored_labels=[]
        )
    assert "Expected image mode to be binary but got mode" in str(exc_info)


def test_predicted_segmentation_validation_format_neg():
    """Check we get an error if the format is not PNG"""
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode="1", size=(20, 20))
        img_path = os.path.join(tempdir, "img.jpg")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

    with pytest.raises(ValueError) as exc_info:
        PredictedSegmentation(
            shape=img_bytes, image=Image(uri="uri"), scored_labels=[]
        )
    assert "Expected image format PNG but got" in str(exc_info)
