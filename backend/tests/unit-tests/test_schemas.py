import os
from base64 import b64encode
from tempfile import TemporaryDirectory

import PIL.Image
import pytest

from velour_api.schemas import (
    DetectionBase,
    Image,
    Label,
    PredictedSegmentation,
)


def test_ground_truth_detection_validation_pos(img: Image):
    boundary = [[1, 1], [2, 2], [0, 4]]
    det = DetectionBase(
        boundary=boundary,
        labels=[Label(key="class", value="a")],
        image=img,
    )
    assert det.boundary == [tuple(pt) for pt in boundary]


def test_ground_truth_detection_validation_neg(img: Image):
    boundary = [[1, 1], [2, 2]]
    with pytest.raises(ValueError) as exc_info:
        DetectionBase(
            boundary=boundary,
            labels=[Label(key="class", value="a")],
            image=img,
        )
    assert "must be composed of at least three points" in str(exc_info)


def _create_b64_mask(mode: str, ext: str) -> str:
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode=mode, size=(20, 20))
        img_path = os.path.join(tempdir, f"img.{ext}")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        return b64encode(img_bytes).decode()


def test_predicted_segmentation_validation_pos(img: Image):
    base64_mask = _create_b64_mask(mode="1", ext="png")

    pred_seg = PredictedSegmentation(
        base64_mask=base64_mask, image=img, scored_labels=[], is_instance=True
    )
    assert pred_seg.base64_mask == base64_mask


def test_predicted_segmentation_validation_mode_neg(img: Image):
    """Check we get an error if the mode is not binary"""
    base64_mask = _create_b64_mask(mode="RGB", ext="png")

    with pytest.raises(ValueError) as exc_info:
        PredictedSegmentation(
            base64_mask=base64_mask,
            image=img,
            scored_labels=[],
            is_instance=False,
        )
    assert "Expected image mode to be binary but got mode" in str(exc_info)


def test_predicted_segmentation_validation_format_neg(img: Image):
    """Check we get an error if the format is not PNG"""
    base64_mask = _create_b64_mask(mode="1", ext="jpg")

    with pytest.raises(ValueError) as exc_info:
        PredictedSegmentation(
            base64_mask=base64_mask,
            image=img,
            scored_labels=[],
            is_instance=True,
        )
    assert "Expected image format PNG but got" in str(exc_info)
