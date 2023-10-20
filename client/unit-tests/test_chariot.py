from dataclasses import dataclass

import pytest

from velour import enums
from velour.integrations.chariot.datasets import (
    _parse_groundtruth_from_evaluation_manifest,
)
from velour.integrations.chariot.models import (
    _parse_chariot_detect_image_object_detection,
    _parse_chariot_predict_image_classification,
    _parse_chariot_predict_proba_image_classification,
)
from velour.schemas import BoundingBox, ImageMetadata, Point

chariot_integration = pytest.importorskip("velour.integrations.chariot")


@pytest.fixture
def img_clf_manifest():
    manifest = [
        {
            "datum_id": "1",
            "path": "s3://img1.jpg",
            "annotations": [{"attributes": {}, "class_label": "dog"}],
        },
        {
            "datum_id": "2",
            "path": "s3://img2.jpg",
            "annotations": [{"attributes": {}, "class_label": "cat"}],
        },
    ]
    assert len(manifest) == 2
    return manifest


@pytest.fixture
def obj_det_manifest():
    manifest = [
        {
            "datum_id": "1",
            "path": "a/b/d/img1.png",
            "annotations": [
                {
                    "attributes": {},
                    "class_label": "dog",
                    "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150},
                },
                {
                    "attributes": {},
                    "class_label": "person",
                    "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110},
                },
            ],
        },
        {
            "datum_id": "2",
            "path": "a/b/d/img2.png",
            "annotations": [
                {
                    "attributes": {},
                    "class_label": "cat",
                    "bbox": {
                        "xmin": 500,
                        "ymin": 220,
                        "xmax": 530,
                        "ymax": 260,
                    },
                }
            ],
        },
        {
            "datum_id": "3",
            "path": "a/b/d/img3.png",
            "annotations": [{"attributes": {}}],
        },
    ]
    assert len(manifest) == 3
    return manifest


@pytest.fixture
def img_seg_manifest():
    manifest = [
        {
            "datum_id": "1",
            "path": "a/b/c/img1.png",
            "annotations": [
                {
                    "attributes": {},
                    "class_label": "dog",
                    "contours": [
                        [
                            {"x": 10.0, "y": 15.5},
                            {"x": 20.9, "y": 50.2},
                            {"x": 25.9, "y": 28.4},
                        ]
                    ],
                }
            ],
        },
        {
            "datum_id": "2",
            "path": "a/b/c/img4.png",
            "annotations": [
                {
                    "attributes": {},
                    "class_label": "car",
                    "contours": [
                        [
                            {"x": 97.2, "y": 40.2},
                            {"x": 33.33, "y": 44.3},
                            {"x": 10.9, "y": 18.7},
                        ],
                        [
                            {"x": 60.0, "y": 15.5},
                            {"x": 70.9, "y": 50.2},
                            {"x": 75.9, "y": 28.4},
                        ],
                    ],
                }
            ],
        },
    ]
    assert len(manifest) == 2
    return manifest


""" Dataset """


def _test_img_clf_manifest(groundtruths):
    assert len(groundtruths) == 2

    # check img 1

    gt = groundtruths[0]
    assert gt.datum.uid == "1"
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "dog", None)
    assert gt.datum.uid == "1"

    # check img 2

    gt = groundtruths[1]
    assert gt.datum.uid == "2"
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "cat", None)
    assert gt.datum.uid == "2"


def _test_obj_det_manifest(groundtruths):
    assert len(groundtruths) == 3

    # Item 1
    gt = groundtruths[0]
    assert gt.datum.uid == "1"
    assert len(gt.annotations) == 2

    # Item 1.a
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "dog", None)
    assert gt.annotations[0].polygon is None
    assert gt.annotations[0].bounding_box == BoundingBox.from_extrema(
        xmin=16, ymin=130, xmax=70, ymax=150
    )

    # Item 1.b
    assert len(gt.annotations[1].labels) == 1
    assert gt.annotations[1].labels[0].tuple() == (
        "class_label",
        "person",
        None,
    )
    assert gt.annotations[1].polygon is None
    assert gt.annotations[1].bounding_box == BoundingBox.from_extrema(
        xmin=89, ymin=10, xmax=97, ymax=110
    )

    # Item 2
    gt = groundtruths[1]
    assert gt.datum.uid == "2"
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "cat", None)
    assert gt.annotations[0].polygon is None
    assert gt.annotations[0].bounding_box == BoundingBox.from_extrema(
        xmin=500, ymin=220, xmax=530, ymax=260
    )


def _test_img_seg_manifest(groundtruths):
    assert len(groundtruths) == 2

    # Item 1
    gt = groundtruths[0]
    assert gt.datum.uid == "1"
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "dog", None)

    assert gt.annotations[0].polygon.boundary.points == [
        Point(10.0, 15.5),
        Point(20.9, 50.2),
        Point(25.9, 28.4),
    ]
    assert gt.annotations[0].polygon.holes is None

    # Item 2
    gt = groundtruths[1]
    assert gt.datum.uid == "2"
    assert len(gt.annotations) == 1
    assert len(gt.annotations[0].labels) == 1
    assert gt.annotations[0].labels[0].tuple() == ("class_label", "car", None)

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


def test__parse_groundtruth(
    img_clf_manifest: list, img_seg_manifest: list, obj_det_manifest: list
):
    # mock chariot supported types
    @dataclass
    class SupportedTaskTypes:
        image_classification = False
        image_segmentation = False
        object_detection = False
        text_sentiment = False
        text_summarization = False
        text_token_classification = False
        text_translation = False

    # mock chariot dataset version
    @dataclass
    class DatasetVersion:
        supported_task_types = SupportedTaskTypes

    dsv = DatasetVersion

    # Image classification
    dsv.supported_task_types.image_classification = True
    _test_img_clf_manifest(
        [
            _parse_groundtruth_from_evaluation_manifest(
                dsv,
                manifest_datum,
            )
            for manifest_datum in img_clf_manifest
        ]
    )
    dsv.supported_task_types.image_classification = False

    # Object Detection
    dsv.supported_task_types.object_detection = True
    _test_obj_det_manifest(
        [
            _parse_groundtruth_from_evaluation_manifest(dsv, manifest_datum)
            for manifest_datum in obj_det_manifest
        ]
    )
    dsv.supported_task_types.object_detection = False

    # Image Semantic Segmentation
    dsv.supported_task_types.image_segmentation = True
    _test_img_seg_manifest(
        [
            _parse_groundtruth_from_evaluation_manifest(
                dsv,
                manifest_datum,
            )
            for manifest_datum in img_seg_manifest
        ]
    )
    dsv.supported_task_types.image_segmentation = False


""" Model """


@pytest.fixture
def img_clf_prediction():
    labels = {"dog": 0, "cat": 1, "elephant": 2}
    pred = ["dog"]
    return pred, labels


@pytest.fixture
def img_clf_prediction_proba():
    labels = {"dog": 0, "cat": 1, "elephant": 2}
    scores = [[0.2, 0.5, 0.3]]
    return scores, labels


@pytest.fixture
def obj_det_prediction():
    return [
        {
            "num_detections": 2,
            "detection_classes": [
                "person",
                "car",
            ],
            "detection_boxes": [
                [
                    151,
                    118,
                    377,
                    197,
                ],
                [
                    94,
                    266,
                    419,
                    352,
                ],
            ],
            "detection_scores": ["0.99", "0.97"],
        }
    ]


def test__parse_chariot_predict_image_classification(
    img_clf_prediction,
):
    chariot_classifications, chariot_labels = img_clf_prediction

    datum = ImageMetadata(uid="", width=1000, height=2000).to_datum()

    velour_classifications = _parse_chariot_predict_image_classification(
        datum,
        chariot_labels,
        chariot_classifications,
    )

    assert len(velour_classifications.annotations) == 1
    assert (
        velour_classifications.annotations[0].task_type
        == enums.TaskType.CLASSIFICATION
    )
    assert velour_classifications.datum == datum

    # validate label key set
    assert set(
        [
            scored_label.key
            for det in velour_classifications.annotations
            for scored_label in det.labels
        ]
    ) == {"class_label"}

    # validate label value set
    assert set(
        [
            scored_label.value
            for det in velour_classifications.annotations
            for scored_label in det.labels
        ]
    ) == {"dog", "cat", "elephant"}

    # validate scores
    for scored_label in velour_classifications.annotations[0].labels:
        if scored_label.value == chariot_classifications[0]:
            assert scored_label.score == 1.0
        else:
            assert scored_label.score == 0.0


def test__parse_chariot_predict_proba_image_classification(
    img_clf_prediction_proba,
):
    chariot_classifications, chariot_labels = img_clf_prediction_proba

    datum = ImageMetadata(uid="", width=1000, height=2000).to_datum()

    velour_classifications = _parse_chariot_predict_proba_image_classification(
        datum,
        chariot_labels,
        chariot_classifications,
    )

    assert len(velour_classifications.annotations) == 1
    assert (
        velour_classifications.annotations[0].task_type
        == enums.TaskType.CLASSIFICATION
    )
    assert velour_classifications.datum == datum

    # validate label key set
    assert set(
        [
            scored_label.key
            for det in velour_classifications.annotations
            for scored_label in det.labels
        ]
    ) == {"class_label"}

    # validate label value set
    assert set(
        [
            scored_label.value
            for det in velour_classifications.annotations
            for scored_label in det.labels
        ]
    ) == {"dog", "cat", "elephant"}

    # validate scores
    for scored_label in velour_classifications.annotations[0].labels:
        idx = chariot_labels[scored_label.value]
        assert chariot_classifications[0][idx] == scored_label.score


def test__parse_chariot_detect_image_object_detection(
    obj_det_prediction,
):
    datum = ImageMetadata(uid="", width=1000, height=2000).to_datum()

    # test parsing
    velour_detections = _parse_chariot_detect_image_object_detection(
        datum, obj_det_prediction
    )

    assert set(
        [
            scored_label.key
            for det in velour_detections.annotations
            for scored_label in det.labels
        ]
    ) == {"class_label"}

    assert set(
        [
            scored_label.value
            for det in velour_detections.annotations
            for scored_label in det.labels
        ]
    ) == {"person", "car"}

    chariot_detection_boxes = obj_det_prediction[0]["detection_boxes"]
    for i, velour_det in enumerate(velour_detections.annotations):
        assert [
            velour_det.bounding_box.ymin,
            velour_det.bounding_box.xmin,
            velour_det.bounding_box.ymax,
            velour_det.bounding_box.xmax,
        ] in chariot_detection_boxes
        assert velour_det.polygon is None
