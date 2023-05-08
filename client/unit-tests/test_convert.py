import json

import PIL.Image

from velour.convert import (
    chariot_detections_to_velour,
    chariot_parse_image_classification_annotation,
    chariot_parse_image_segmentation_annotation,
    chariot_parse_object_detection_annotation,
    coco_rle_to_mask,
)
from velour.data_types import BoundingBox, Image, Point


def test_chariot_detections_to_velour():
    dets = {
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

    velour_dets = chariot_detections_to_velour(
        dets, Image(uid="", width=10, height=100)
    )

    assert len(velour_dets) == 2
    assert [
        scored_label.label.key
        for det in velour_dets
        for scored_label in det.scored_labels
    ] == ["class", "class"]
    assert set(
        [
            scored_label.label.value
            for det in velour_dets
            for scored_label in det.scored_labels
        ]
    ) == {"person", "car"}

    for i, velour_det in enumerate(velour_dets):
        assert dets["detection_boxes"][i] == [
            velour_det.bbox.ymin,
            velour_det.bbox.xmin,
            velour_det.bbox.ymax,
            velour_det.bbox.xmax,
        ]

        assert velour_det.boundary is None


def test_coco_rle_to_mask():
    h, w = 4, 6
    coco_rle_seg_dict = {"counts": [10, 4, 3, 7], "size": (h, w)}

    mask = coco_rle_to_mask(coco_rle_seg_dict=coco_rle_seg_dict)
    expected_mask = [
        [False, False, False, True, False, True],
        [False, False, False, True, True, True],
        [False, False, True, False, True, True],
        [False, False, True, False, True, True],
    ]

    assert mask.sum() == 4 + 7
    assert mask.tolist() == expected_mask

    img = PIL.Image.fromarray(mask)

    assert img.width == w
    assert img.height == h


def test_chariot_integration_image_classification():
    jsonl = '{"path": "a/b/c/img1.png", "annotations": [{"class_label": "dog"}]}\n{"path": "a/b/c/img2.png", "annotations": [{"class_label": "cat"}]}'
    jsonl = jsonl.split("\n")

    chariot_dataset = []
    for line in jsonl:
        chariot_dataset.append(json.loads(line))

    assert len(chariot_dataset) == 2

    # Item 1
    velour_datum = chariot_parse_image_classification_annotation(
        chariot_dataset[0]
    )
    assert len(velour_datum) == 1
    velour_datum = velour_datum[0]
    assert len(velour_datum.labels) == 1
    assert velour_datum.labels[0].tuple() == ("class_label", "dog")
    assert velour_datum.image.uid == "img1"
    assert velour_datum.image.height is None
    assert velour_datum.image.width is None
    assert velour_datum.image.frame is None

    # Item 2
    velour_datum = chariot_parse_image_classification_annotation(
        chariot_dataset[1]
    )
    assert len(velour_datum) == 1
    velour_datum = velour_datum[0]
    assert len(velour_datum.labels) == 1
    assert velour_datum.labels[0].tuple() == ("class_label", "cat")
    assert velour_datum.image.uid == "img2"
    assert velour_datum.image.height is None
    assert velour_datum.image.width is None
    assert velour_datum.image.frame is None


def test_chariot_integration_image_segmentation():
    jsonl = '{"path": "a/b/c/img1.png", "annotations": [{"class_label": "dog", "contour": [[{"x": 10.0, "y": 15.5}, {"x": 20.9, "y": 50.2}, {"x": 25.9, "y": 28.4}]]}]}\n{"path": "a/b/c/img4.png", "annotations": [{"class_label": "car", "contour": [[{"x": 97.2, "y": 40.2}, {"x": 33.33, "y": 44.3}, {"x": 10.9, "y": 18.7}], [{"x": 60.0, "y": 15.5}, {"x": 70.9, "y": 50.2}, {"x": 75.9, "y": 28.4}]]}]}'
    jsonl = jsonl.split("\n")

    chariot_dataset = []
    for line in jsonl:
        chariot_dataset.append(json.loads(line))

    assert len(chariot_dataset) == 2

    # Item 1
    velour_datum = chariot_parse_image_segmentation_annotation(
        chariot_dataset[0]
    )
    assert len(velour_datum) == 1
    velour_datum = velour_datum[0]
    assert len(velour_datum.labels) == 1
    assert velour_datum.labels[0].tuple() == ("class_label", "dog")
    assert velour_datum.image.uid == "img1"
    assert velour_datum.image.height is None
    assert velour_datum.image.width is None
    assert velour_datum.image.frame is None
    assert len(velour_datum.shape) == 1
    assert velour_datum.shape[0].polygon.points == [
        Point(10.0, 15.5),
        Point(20.9, 50.2),
        Point(25.9, 28.4),
    ]
    assert velour_datum.shape[0].hole is None

    # Item 2
    velour_datum = chariot_parse_image_segmentation_annotation(
        chariot_dataset[1]
    )
    assert len(velour_datum) == 1
    velour_datum = velour_datum[0]
    assert len(velour_datum.labels) == 1
    assert velour_datum.labels[0].tuple() == ("class_label", "car")
    assert velour_datum.image.uid == "img4"
    assert velour_datum.image.height is None
    assert velour_datum.image.width is None
    assert velour_datum.image.frame is None
    assert len(velour_datum.shape) == 1
    assert velour_datum.shape[0].polygon.points == [
        Point(97.2, 40.2),
        Point(33.33, 44.3),
        Point(10.9, 18.7),
    ]
    assert velour_datum.shape[0].hole.points == [
        Point(60.0, 15.5),
        Point(70.9, 50.2),
        Point(75.9, 28.4),
    ]


def test_chariot_integration_object_detection():
    jsonl = '{"path": "a/b/d/img1.png", "annotations": [{"class_label": "dog", "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150}}, {"class_label": "person", "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110}}]}\n{"path": "a/b/d/img2.png", "annotations": [{"class_label": "cat", "bbox": {"xmin": 500, "ymin": 220, "xmax": 530, "ymax": 260}}]}\n{"path": "a/b/d/img3.png", "annotations": []}'
    jsonl = jsonl.split("\n")

    chariot_dataset = []
    for line in jsonl:
        chariot_dataset.append(json.loads(line))

    assert len(chariot_dataset) == 3

    # Item 1 - Multiple objects of interest
    velour_datum = chariot_parse_object_detection_annotation(
        chariot_dataset[0]
    )
    assert len(velour_datum) == 2

    assert len(velour_datum[0].labels) == 1
    assert velour_datum[0].labels[0].tuple() == ("class_label", "dog")
    assert velour_datum[0].image.uid == "img1"
    assert velour_datum[0].image.height is None
    assert velour_datum[0].image.width is None
    assert velour_datum[0].image.frame is None
    assert velour_datum[0].boundary is None
    assert velour_datum[0].bbox == BoundingBox(16, 130, 70, 150)

    assert len(velour_datum[1].labels) == 1
    assert velour_datum[1].labels[0].tuple() == ("class_label", "person")
    assert velour_datum[1].image.uid == "img1"
    assert velour_datum[1].image.height is None
    assert velour_datum[1].image.width is None
    assert velour_datum[1].image.frame is None
    assert velour_datum[1].boundary is None
    assert velour_datum[1].bbox == BoundingBox(89, 10, 97, 110)

    # Item 2 - Single object of interest
    velour_datum = chariot_parse_object_detection_annotation(
        chariot_dataset[1]
    )
    assert len(velour_datum) == 1
    velour_datum = velour_datum[0]
    assert len(velour_datum.labels) == 1
    assert velour_datum.labels[0].tuple() == ("class_label", "cat")
    assert velour_datum.image.uid == "img2"
    assert velour_datum.image.height is None
    assert velour_datum.image.width is None
    assert velour_datum.image.frame is None
    assert velour_datum.boundary is None
    assert velour_datum.bbox == BoundingBox(500, 220, 530, 260)

    # Item 3 - No object of interest
    velour_datum = chariot_parse_object_detection_annotation(
        chariot_dataset[2]
    )
    assert len(velour_datum) == 0


# def test_chariot_load_dataset():

#     from chariot.client import connect
#     from chariot.datasets.dataset import get_datasets_in_project

#     # List available datasets in project
#     project_name = "Global"
#     datasets = get_datasets_in_project(
#         limit=25, offset=0, project_name=project_name
#     )

#     dslu = {}
#     print("Datasets")
#     for i in range(len(datasets)):
#         dslu[str(datasets[i].name).strip()] = datasets[i]
#         print(" " + str(i) + ": " + datasets[i].name)

#     idx = int(input())
#     dsv = datasets[idx].versions[0]

#     retval = chariot_ds_to_velour_ds(dsv, "Test")
