import os
from base64 import b64encode
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import pytest
from pydantic import ValidationError

from velour_api.enums import JobStatus
from velour_api.schemas import (
    ConfusionMatrix,
    Dataset,
    DatumMetadatum,
    DatumTypes,
    GroundTruthDetection,
    GroundTruthSegmentation,
    Image,
    Job,
    Label,
    LabelDistribution,
    Model,
    PredictedSegmentation,
    ScoredLabel,
    ScoredLabelDistribution,
)


def _create_b64_mask(mode: str, ext: str, size=(20, 20)) -> str:
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode=mode, size=size)
        img_path = os.path.join(tempdir, f"img.{ext}")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        return b64encode(img_bytes).decode()


def test_ground_truth_detection_validation_pos(img: Image):
    boundary = [[1, 1], [2, 2], [0, 4]]
    det = GroundTruthDetection(
        boundary=boundary,
        labels=[Label(key="class", value="a")],
        image=img,
    )
    assert det.boundary == [tuple(pt) for pt in boundary]

    det = GroundTruthDetection(
        bbox=(1, 2, 3, 4),
        labels=[Label(key="class", value="a")],
        image=img,
    )


def test_ground_truth_detection_validation_neg(img: Image):
    boundary = [(1, 1), (2, 2)]

    with pytest.raises(ValueError) as exc_info:
        GroundTruthDetection(
            boundary=boundary,
            labels=[Label(key="class", value="a")],
            image=img,
        )
    assert "must be composed of at least three points" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        GroundTruthDetection(
            boundary=boundary + [(3, 4)],
            bbox=(1, 2, 3, 4),
            labels=[Label(key="class", value="a")],
            image=img,
        )
    assert "Must have exactly one of boundary or bbox" in str(exc_info)


def test_ground_truth_segmentation_validation(img: Image):
    mask = _create_b64_mask(mode="1", ext="PNG")
    with pytest.raises(ValidationError) as exc_info:
        GroundTruthSegmentation(
            shape=mask, image=img, labels=[], is_instance=True
        )
    assert "Expected mask and image to have the same size" in str(exc_info)

    mask = _create_b64_mask(mode="1", ext="PNG", size=(img.width, img.height))
    assert GroundTruthSegmentation(
        shape=mask, image=img, labels=[], is_instance=True
    )


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


def test_dataset_validation():
    with pytest.raises(ValueError) as exc_info:
        Dataset(name="name", href="not valid", type=DatumTypes.IMAGE)
    assert "`href` must" in str(exc_info)

    assert Dataset(name="name", href="http://a.com", type=DatumTypes.IMAGE)


def test_model_validation():
    with pytest.raises(ValueError) as exc_info:
        Model(name="name", href="not valid")
    assert "`href` must" in str(exc_info)

    assert Model(name="name", href="http://a.com", type=DatumTypes.IMAGE)


def test_eval_job():
    job = Job()
    # check that job got a uid of the right form
    assert isinstance(job.uid, str)
    assert len(job.uid.split("-")) == 5

    assert job.status == JobStatus.PENDING


def test_confusion_matrix(cm: ConfusionMatrix):
    np.testing.assert_array_equal(
        cm.matrix, np.array([[1, 1, 1], [0, 1, 0], [0, 2, 0]])
    )

    assert cm.matrix[cm.label_map["class1"], cm.label_map["class2"]] == 0
    assert cm.matrix[cm.label_map["class1"], cm.label_map["class1"]] == 1


def test_datum_metadatum_validation():
    md = DatumMetadatum(name="meta name", value=1)
    assert md.value == 1

    md = DatumMetadatum(name="meta name", value="a string")
    assert md.value == "a string"

    md = DatumMetadatum(name="meta name", value={"a": 1, "b": "c"})
    assert md.value == {"a": 1, "b": "c"}

    # check we get a JSON serialization error since a set
    # is not JSON serializable
    with pytest.raises(TypeError) as exc_info:
        DatumMetadatum(name="meta name", value={"a": {1, 2}})

    assert "type set is not JSON serializable" in str(exc_info)


def test_label_types():
    l1 = Label(key="key1", value="value1")
    l2 = Label(key="key2", value="value2")
    l3 = Label(key="key1", value="value1")

    # test equality
    assert l1 != l2
    assert l1 == l3
    l3.value = "value3"
    assert l1 != l3

    # test as key
    d = {l3: "testing"}
    assert d[l3] == "testing"

    assert l1.model_dump_json() == '{"key":"key1","value":"value1"}'
    assert (
        ScoredLabel(label=l1, score=0.5).model_dump_json()
        == '{"label":{"key":"key1","value":"value1"},"score":0.5}'
    )
    assert (
        LabelDistribution(label=l1, count=1).model_dump_json()
        == '{"label":{"key":"key1","value":"value1"},"count":1}'
    )
    assert (
        ScoredLabelDistribution(
            label=l1, scores=[0.1, 0.9], count=2
        ).model_dump_json()
        == '{"label":{"key":"key1","value":"value1"},"scores":[0.1,0.9],"count":2}'
    )
