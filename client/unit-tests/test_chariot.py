import json
from dataclasses import dataclass

import pytest

from velour.schemas import BoundingBox, GroundTruth, Image, Point

chariot_dsv = pytest.importorskip("chariot.datasets.dataset_version")
chariot_swagger = pytest.importorskip(
    "chariot._swagger.datasets.models.output_dataset_version_summary"
)
chariot_integration = pytest.importorskip("velour.integrations.chariot")


@pytest.fixture
def img_clf_ds():
    jsonl = '{"path": "a/b/c/img1.png", "annotations": [{"class_label": "dog"}]}\n{"path": "a/b/c/img2.png", "annotations": [{"class_label": "cat"}]}'
    ds = [json.loads(line) for line in jsonl.split("\n")]
    assert len(ds) == 2
    return ds


@pytest.fixture
def img_seg_ds():
    jsonl = '{"path": "a/b/c/img1.png", "annotations": [{"class_label": "dog", "contours": [[{"x": 10.0, "y": 15.5}, {"x": 20.9, "y": 50.2}, {"x": 25.9, "y": 28.4}]]}]}\n{"path": "a/b/c/img4.png", "annotations": [{"class_label": "car", "contours": [[{"x": 97.2, "y": 40.2}, {"x": 33.33, "y": 44.3}, {"x": 10.9, "y": 18.7}], [{"x": 60.0, "y": 15.5}, {"x": 70.9, "y": 50.2}, {"x": 75.9, "y": 28.4}]]}]}'
    ds = [json.loads(line) for line in jsonl.split("\n")]
    assert len(ds) == 2
    return ds


@pytest.fixture
def obj_det_ds():
    jsonl = '{"path": "a/b/d/img1.png", "annotations": [{"class_label": "dog", "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150}}, {"class_label": "person", "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110}}]}\n{"path": "a/b/d/img2.png", "annotations": [{"class_label": "cat", "bbox": {"xmin": 500, "ymin": 220, "xmax": 530, "ymax": 260}}]}\n{"path": "a/b/d/img3.png", "annotations": []}'
    ds = [json.loads(line) for line in jsonl.split("\n")]
    assert len(ds) == 3
    return ds


def _test_img_clf_ds(velour_dataset):

    assert len(velour_dataset) == 2

    gt = velour_dataset[0]
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "dog")
    assert gt.datum.uid == "img1"

    image = Image.from_datum(gt.datum)
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0

    gt = velour_dataset[1]
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "cat")
    assert gt.datum.uid == "img2"

    image = Image.from_datum(gt.datum)
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0


def _test_img_seg_ds(velour_dataset):

    assert len(velour_dataset) == 2

    # Item 1
    gt = velour_dataset[0]
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "dog")

    image = Image.from_datum(gt.datum)
    assert image.uid == "img1"
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0

    assert gt.annotations[0].polygon.boundary.points == [
        Point(10.0, 15.5),
        Point(20.9, 50.2),
        Point(25.9, 28.4),
    ]
    assert gt.annotations[0].polygon.holes is None

    # Item 2
    gt = velour_dataset[1]
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "car")

    image = Image.from_datum(gt.datum)
    assert image.uid == "img4"
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0

    assert gt.annotations[0].polygon.boundary.points == [
        Point(97.2, 40.2),
        Point(33.33, 44.3),
        Point(10.9, 18.7),
    ]
    assert gt.annotations[0].polygon.holes[0].points == [
        Point(60.0, 15.5),
        Point(70.9, 50.2),
        Point(75.9, 28.4),
    ]


def _test_obj_det_ds(velour_dataset):

    assert len(velour_dataset) == 3

    # Item 1
    gt = velour_dataset[0]
    assert len(gt.annotations) == 2

    image = Image.from_datum(gt.datum)
    assert image.uid == "img1"
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0

    # Item 1.a
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "dog")
    assert gt.annotations[0].polygon is None
    assert gt.annotations[0].bounding_box == BoundingBox.from_extrema(
        xmin=16, ymin=130, xmax=70, ymax=150
    )

    # Item 1.b
    assert len(gt.annotations[1].labels) == 1
    assert gt.annotations[1].labels[0].tuple() == ("class", "person")
    assert gt.annotations[1].polygon is None
    assert gt.annotations[1].bounding_box == BoundingBox.from_extrema(
        xmin=89, ymin=10, xmax=97, ymax=110
    )

    # Item 2
    gt = velour_dataset[1]
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class", "cat")

    image = Image.from_datum(gt.datum)
    assert image.uid == "img2"
    assert image.height == -1
    assert image.width == -1
    assert image.frame == 0

    assert gt.annotations[0].polygon is None
    assert gt.annotations[0].bounding_box == BoundingBox.from_extrema(
        xmin=500, ymin=220, xmax=530, ymax=260
    )


def test__parse_image_classification_groundtruths(img_clf_ds: str):
    chariot_dataset = img_clf_ds
    item1 = chariot_integration._parse_image_classification_groundtruths(
        chariot_dataset[0]
    )
    assert isinstance(item1, GroundTruth)
    item2 = chariot_integration._parse_image_classification_groundtruths(
        chariot_dataset[1]
    )
    assert isinstance(item2, GroundTruth)
    velour_dataset = [item1, item2]
    _test_img_clf_ds(velour_dataset)


def test__parse_image_segmentation_groundtruths(img_seg_ds: str):
    chariot_dataset = img_seg_ds
    item1 = chariot_integration._parse_image_segmentation_groundtruths(
        chariot_dataset[0]
    )
    assert isinstance(item1, GroundTruth)
    item2 = chariot_integration._parse_image_segmentation_groundtruths(
        chariot_dataset[1]
    )
    assert isinstance(item2, GroundTruth)
    velour_dataset = [item1, item2]
    _test_img_seg_ds(velour_dataset=velour_dataset)


def test__parse_object_detection_groundtruths(obj_det_ds: str):
    chariot_dataset = obj_det_ds

    # Item 1 - Multiple objects of interest
    item1 = chariot_integration._parse_object_detection_groundtruths(
        chariot_dataset[0]
    )
    assert isinstance(item1, GroundTruth)

    # Item 2 - Single object of interest
    item2 = chariot_integration._parse_object_detection_groundtruths(
        chariot_dataset[1]
    )
    assert isinstance(item2, GroundTruth)

    # Item 3 - No object of interest
    item3 = chariot_integration._parse_object_detection_groundtruths(
        chariot_dataset[2]
    )
    assert isinstance(item3, GroundTruth)

    velour_dataset = [item1, item2, item3]
    _test_obj_det_ds(velour_dataset=velour_dataset)


def test__parse_dataset_version_manifest(
    img_clf_ds: str, img_seg_ds: str, obj_det_ds: str
):
    @dataclass
    class supported_task_types:
        image_classification = False
        image_segmentation = False
        object_detection = False
        text_sentiment = False
        text_summarization = False
        text_token_classification = False
        text_translation = False

    chariot_task_type = supported_task_types

    # Image classification
    chariot_task_type.image_classification = True
    _test_img_clf_ds(
        chariot_integration._parse_chariot_groundtruths(
            img_clf_ds, chariot_task_type
        )
    )
    chariot_task_type.image_classification = False

    # Image Semantic Segmentation
    chariot_task_type.image_segmentation = True
    _test_img_seg_ds(
        chariot_integration._parse_chariot_groundtruths(
            img_seg_ds, chariot_task_type
        )
    )
    chariot_task_type.image_segmentation = False

    # Object Detection
    chariot_task_type.object_detection = True
    _test_obj_det_ds(
        chariot_integration._parse_chariot_groundtruths(
            obj_det_ds, chariot_task_type
        )
    )
    chariot_task_type.object_detection = False


def test_parse_chariot_image_classifications():
    try:
        chariot_integration.parse_chariot_image_classifications(None, None)
    except NotImplementedError:
        pass


def test_parse_chariot_image_segmentations():
    try:
        chariot_integration.parse_chariot_image_segmentations(None, None)
    except NotImplementedError:
        pass


def _test_parse_chariot_object_detections(
    chariot_detections, velour_detections
):
    annotations = []
    for datum in velour_detections:
        annotations.extend([annotation for annotation in datum.annotations])

    assert set(
        [
            scored_label.label.key
            for det in annotations
            for scored_label in det.scored_labels
        ]
    ) == {"class"}

    assert set(
        [
            scored_label.label.value
            for det in annotations
            for scored_label in det.scored_labels
        ]
    ) == {"person", "car"}

    chariot_detection_boxes = chariot_detections["detection_boxes"]
    for i, velour_det in enumerate(annotations):
        assert [
            velour_det.bounding_box.ymin,
            velour_det.bounding_box.xmin,
            velour_det.bounding_box.ymax,
            velour_det.bounding_box.xmax,
        ] in chariot_detection_boxes
        assert velour_det.polygon is None


def test_parse_chariot_object_detections():

    chariot_detections = {
        "num_detections": 2,
        "detection_classes": [
            "person",
            "car",
        ],
        "detection_boxes": [
            [
                151.2235107421875,
                118.97279357910156,
                377.8422546386719,
                197.98605346679688,
            ],
            [
                94.09261322021484,
                266.5445556640625,
                419.3203430175781,
                352.9458923339844,
            ],
        ],
        "detection_scores": ["0.99932003", "0.99895525"],
    }
    image = Image(uid="", width=10, height=100)

    # Test unwrapped input
    velour_detections = chariot_integration.parse_chariot_object_detections(
        chariot_detections, image
    )
    _test_parse_chariot_object_detections(
        chariot_detections, velour_detections
    )

    # Test unwrapped det, image list
    velour_detections = chariot_integration.parse_chariot_object_detections(
        chariot_detections, [image]
    )
    _test_parse_chariot_object_detections(
        chariot_detections, velour_detections
    )

    # Test det list, unwrapped image
    velour_detections = chariot_integration.parse_chariot_object_detections(
        [chariot_detections], image
    )
    _test_parse_chariot_object_detections(
        chariot_detections, velour_detections
    )

    # Test wrapped inputs
    velour_detections = chariot_integration.parse_chariot_object_detections(
        [chariot_detections], [image]
    )
    _test_parse_chariot_object_detections(
        chariot_detections, velour_detections
    )

    # Test multiple inputs
    velour_detections = chariot_integration.parse_chariot_object_detections(
        [chariot_detections, chariot_detections], [image, image]
    )
    for image_detections in velour_detections:
        _test_parse_chariot_object_detections(
            chariot_detections, [image_detections]
        )

    # Test mismatch size
    try:
        velour_detections = (
            chariot_integration.parse_chariot_object_detections(
                [chariot_detections, chariot_detections], [image, image]
            )
        )
        for image_detections in velour_detections:
            _test_parse_chariot_object_detections(
                chariot_detections, [image_detections]
            )
    except AssertionError as e:
        assert e.args[0] == "length mismatch"
