import numpy
import pytest
import torch

from velour.data_types import (
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
)
from velour.integrations.yolo import (
    _convert_yolo_segmentation,
    parse_image_classification,
    parse_image_segmentation,
    parse_object_detection,
)


class Boxes(object):
    def __init__(self, boxes: torch.Tensor, orig_shape: tuple):
        self.data = boxes
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]


class Masks(object):
    def __init__(
        self,
        masks: torch.Tensor,
        orig_shape: tuple,
    ):
        self.data = masks
        self.orig_shape = orig_shape


class Results(object):
    def __init__(
        self,
        orig_img: torch.Tensor,
        path: str,
        names: dict,
        probs: torch.Tensor = None,
        boxes: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names
        self.probs = probs
        self.boxes = (
            Boxes(boxes, self.orig_shape) if boxes is not None else None
        )
        self.masks = (
            Masks(masks, self.orig_shape) if masks is not None else None
        )
        self.keys = []
        self.conf = None

        if probs is not None:
            self.keys = ["probs"]
        elif boxes is not None and masks is not None:
            self.keys = ["boxes", "masks"]

        elif boxes is not None and masks is None:
            self.keys = ["boxes"]
        else:
            raise ValueError("Invalid configuration of simulated results.")


@pytest.fixture
def image():
    return {
        "path": "a/b/c/d.png",
        "uid": "d",
        "height": 1280,
        "width": 960,
        "mask_height": 640,
        "mask_width": 480,
    }


@pytest.fixture
def names():
    return {0: "dog", 1: "cat", 2: "person"}


@pytest.fixture
def bbox1(names):
    return {
        "xmin": 0,
        "ymin": 10,
        "xmax": 20,
        "ymax": 30,
        "class": 0,
        "confidence": 0.54,
    }


@pytest.fixture
def bbox2(names):
    return {
        "xmin": 40,
        "ymin": 50,
        "xmax": 60,
        "ymax": 70,
        "class": 1,
        "confidence": 0.98,
    }


@pytest.fixture
def bbox3(names):
    return {
        "xmin": 80,
        "ymin": 90,
        "xmax": 100,
        "ymax": 110,
        "class": 2,
        "confidence": 0.41,
    }


@pytest.fixture
def bboxes(bbox1, bbox2, bbox3):

    boxes = torch.zeros(3, 6)

    boxes[0][0] = bbox1["xmin"]
    boxes[0][1] = bbox1["ymin"]
    boxes[0][2] = bbox1["xmax"]
    boxes[0][3] = bbox1["ymax"]
    boxes[0][4] = bbox1["confidence"]
    boxes[0][5] = bbox1["class"]

    boxes[1][0] = bbox2["xmin"]
    boxes[1][1] = bbox2["ymin"]
    boxes[1][2] = bbox2["xmax"]
    boxes[1][3] = bbox2["ymax"]
    boxes[1][4] = bbox2["confidence"]
    boxes[1][5] = bbox2["class"]

    boxes[2][0] = bbox3["xmin"]
    boxes[2][1] = bbox3["ymin"]
    boxes[2][2] = bbox3["xmax"]
    boxes[2][3] = bbox3["ymax"]
    boxes[2][4] = bbox3["confidence"]
    boxes[2][5] = bbox3["class"]

    return boxes


@pytest.fixture
def yolo_mask(image):
    x = torch.zeros(image["mask_height"], image["mask_width"], dtype=float)
    x[int(image["mask_height"] / 2) :, int(image["mask_width"] / 2) :] = 1.0
    # One-quarter of the image is colored.
    assert x[x >= 0.5].numel() / x.numel() == 0.25
    return x


@pytest.fixture
def velour_mask(image):
    x = numpy.zeros((image["height"], image["width"]), dtype=numpy.uint8)
    x[int(image["height"] / 2) :, int(image["width"] / 2) :] = 255
    # One-quarter of the image is colored.
    assert x[x >= 128].size / x.size == 0.25
    return x >= 128


def test_parse_image_classification(image, names):

    probs = torch.Tensor([0.82, 0.08, 0.1])

    results = Results(
        orig_img=torch.rand(image["height"], image["width"], 3),
        path=image["path"],
        names=names,
        probs=probs,
    )

    prediction = parse_image_classification(results, image["uid"])[0]

    assert isinstance(prediction, PredictedImageClassification)

    assert prediction.image.uid == image["uid"]
    assert prediction.image.height == image["height"]
    assert prediction.image.width == image["width"]
    assert prediction.image.frame is None

    for i in range(len(prediction.scored_labels)):
        assert prediction.scored_labels[i].label.key == "class_label"
        assert prediction.scored_labels[i].label.value == names[i]
        assert prediction.scored_labels[i].score == probs[i]


def test__convert_yolo_segmentation(image, yolo_mask, velour_mask):
    output = _convert_yolo_segmentation(
        yolo_mask, image["height"], image["width"]
    )
    assert output.shape == velour_mask.shape
    assert (output == velour_mask).all()


def test_parse_image_segmentation(
    image, names, bboxes, yolo_mask, velour_mask
):
    img = torch.rand(image["height"], image["width"], 3)
    masks = torch.stack([yolo_mask, yolo_mask, yolo_mask])

    results = Results(
        orig_img=img,
        path=image["path"],
        names=names,
        boxes=bboxes,
        masks=masks,
    )

    predictions = parse_image_segmentation(results, image["uid"])

    assert len(predictions) == bboxes.size(dim=0)
    for i in range(len(predictions)):
        assert isinstance(predictions[i], PredictedInstanceSegmentation)
        assert predictions[i].image.uid == image["uid"]
        assert predictions[i].image.height == image["height"]
        assert predictions[i].image.width == image["width"]
        assert predictions[i].image.frame is None
        assert predictions[i].scored_labels.label.key == "class_label"
        assert predictions[i].scored_labels.label.value == names[i]
        assert predictions[i].scored_labels.score == bboxes[i][4]
        assert predictions[i].mask.shape == velour_mask.shape
        assert (predictions[i].mask == velour_mask).all()


def test_parse_object_detection(image, bboxes, names):
    img = torch.rand(image["height"], image["width"], 3)

    results = Results(
        orig_img=img, path=image["path"], names=names, boxes=bboxes
    )

    predictions = parse_object_detection(results, image["uid"])

    assert len(predictions) == bboxes.size(dim=0)
    for i in range(len(predictions)):
        assert isinstance(predictions[i], PredictedDetection)
        assert predictions[i].image.uid == image["uid"]
        assert predictions[i].image.height == image["height"]
        assert predictions[i].image.width == image["width"]
        assert predictions[i].image.frame is None
        assert predictions[i].scored_labels.label.key == "class_label"
        assert predictions[i].scored_labels.label.value == names[i]
        assert predictions[i].scored_labels.score == bboxes[i][4]
        assert predictions[i].bbox.xmin == bboxes[i][0]
        assert predictions[i].bbox.ymin == bboxes[i][1]
        assert predictions[i].bbox.xmax == bboxes[i][2]
        assert predictions[i].bbox.ymax == bboxes[i][3]
