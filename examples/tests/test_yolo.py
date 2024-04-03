import numpy
import pytest
from integrations.yolo import (
    _convert_yolo_segmentation,
    parse_yolo_image_classification,
    parse_yolo_image_segmentation,
    parse_yolo_object_detection,
)

from valor import Annotation, Prediction
from valor.metatypes import ImageMetadata


class BoxOnGPU(object):
    def __init__(
        self,
        datum,
    ):
        self.datum = datum

    def cpu(self):
        return self.datum


class Boxes(object):
    def __init__(self, boxes, orig_shape: tuple):
        self.data = boxes
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return [BoxOnGPU(datum) for datum in self.data[:, :4]]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]


class MaskOnGPU(object):
    def __init__(
        self,
        mask,
    ):
        self.mask = mask

    def cpu(self):
        return self.mask


class Masks(object):
    def __init__(
        self,
        masks,
        orig_shape: tuple,
    ):
        self.data = [MaskOnGPU(mask) for mask in masks]
        self.orig_shape = orig_shape


class Results(object):
    def __init__(
        self,
        orig_img,
        path: str,
        names: dict,
        probs=None,
        boxes=None,
        masks=None,
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
    boxes = numpy.zeros((3, 6))

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
    x = numpy.zeros(
        (image["mask_height"], image["mask_width"]), dtype=numpy.float32
    )
    x[int(image["mask_height"] / 2) :, int(image["mask_width"] / 2) :] = 1.0
    # One-quarter of the image is colored.
    assert x[x >= 0.5].size / x.size == 0.25
    return x


@pytest.fixture
def valor_mask(image):
    x = numpy.zeros((image["height"], image["width"]), dtype=numpy.uint8)
    x[int(image["height"] / 2) :, int(image["width"] / 2) :] = 255
    # One-quarter of the image is colored.
    assert x[x >= 128].size / x.size == 0.25
    return x >= 128


def test_parse_yolo_image_classification(image, names):
    probs = numpy.asarray([0.82, 0.08, 0.1])

    results = Results(
        orig_img=numpy.random.rand(image["height"], image["width"], 3),
        path=image["path"],
        names=names,
        probs=probs,
    )

    valor_image = ImageMetadata(
        dataset="dataset",
        uid=image["uid"],
        height=image["height"],
        width=image["width"],
    )

    prediction = parse_yolo_image_classification(
        results, valor_image, label_key="class"
    )

    assert isinstance(prediction, Prediction)

    image_datum = ImageMetadata.from_datum(prediction.datum)
    assert image_datum.uid == image["uid"]
    assert image_datum.height == image["height"]
    assert image_datum.width == image["width"]

    assert len(prediction.annotations) == 1
    annotation = prediction.annotations[0]
    for i in range(len(annotation.labels)):
        assert annotation.labels[i].key == "class"
        assert annotation.labels[i].value == names[i]
        assert annotation.labels[i].score == probs[i]


def test__convert_yolo_segmentation(image, yolo_mask, valor_mask):
    yolo_mask = MaskOnGPU(yolo_mask)
    output = _convert_yolo_segmentation(
        yolo_mask, image["height"], image["width"]
    )
    assert output.shape == valor_mask.shape
    assert (output == valor_mask).all()


def test_parse_yolo_image_segmentation(
    image, names, bboxes, yolo_mask, valor_mask
):
    img = numpy.random.rand(image["height"], image["width"], 3)
    masks = numpy.stack([yolo_mask, yolo_mask, yolo_mask])

    results = Results(
        orig_img=img,
        path=image["path"],
        names=names,
        boxes=bboxes,
        masks=masks,
    )

    valor_image = ImageMetadata(
        dataset="dataset",
        uid=image["uid"],
        height=image["height"],
        width=image["width"],
    )

    prediction = parse_yolo_image_segmentation(
        results, valor_image, label_key="class"
    )

    image_datum = ImageMetadata.from_datum(prediction.datum)
    assert image_datum.uid == image["uid"]
    assert image_datum.height == image["height"]
    assert image_datum.width == image["width"]

    assert isinstance(prediction, Prediction)
    assert len(prediction.annotations) == bboxes.shape[0]
    for i in range(len(prediction.annotations)):
        assert isinstance(prediction.annotations[i], Annotation)
        assert prediction.annotations[i].labels[0].key == "class"
        assert prediction.annotations[i].labels[0].value == names[i]
        assert prediction.annotations[i].labels[0].score == bboxes[i][4]
        assert (
            prediction.annotations[i].raster.to_numpy() == valor_mask
        ).all()


def test_parse_yolo_object_detection(image, bboxes, names):
    img = numpy.random.rand(image["height"], image["width"], 3)

    results = Results(
        orig_img=img, path=image["path"], names=names, boxes=bboxes
    )

    valor_image = ImageMetadata(
        dataset="dataset",
        uid=image["uid"],
        height=image["height"],
        width=image["width"],
    )

    prediction = parse_yolo_object_detection(
        results, valor_image, label_key="class"
    )

    image_datum = ImageMetadata.from_datum(prediction.datum)
    assert image_datum.uid == image["uid"]
    assert image_datum.height == image["height"]
    assert image_datum.width == image["width"]

    assert isinstance(prediction, Prediction)
    assert len(prediction.annotations) == bboxes.shape[0]
    for i in range(len(prediction.annotations)):
        assert isinstance(prediction.annotations[i], Annotation)
        assert prediction.annotations[i].labels[0].key == "class"
        assert prediction.annotations[i].labels[0].value == names[i]
        assert prediction.annotations[i].labels[0].score == bboxes[i][4]
        assert prediction.annotations[i].box.xmin == bboxes[i][0]
        assert prediction.annotations[i].box.ymin == bboxes[i][1]
        assert prediction.annotations[i].box.xmax == bboxes[i][2]
        assert prediction.annotations[i].box.ymax == bboxes[i][3]
