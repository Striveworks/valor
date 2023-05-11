import json

from velour.client import Client, ClientException
from velour.data_types import BoundingBox, Point
from velour.integrations.chariot import (
    chariot_ds_to_velour_ds,
    chariot_parse_image_classification_annotation,
    chariot_parse_image_segmentation_annotation,
    chariot_parse_object_detection_annotation,
)


def test_chariot_parse_image_classification_annotation():
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
    assert velour_datum.image.height == -1
    assert velour_datum.image.width == -1
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
    assert velour_datum.image.height == -1
    assert velour_datum.image.width == -1
    assert velour_datum.image.frame is None


def test_chariot_parse_image_segmentation_annotation():
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
    assert velour_datum.image.height == -1
    assert velour_datum.image.width == -1
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
    assert velour_datum.image.height == -1
    assert velour_datum.image.width == -1
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


def test_chariot_parse_object_detection_annotation():
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
    assert velour_datum[0].image.height == -1
    assert velour_datum[0].image.width == -1
    assert velour_datum[0].image.frame is None
    assert velour_datum[0].boundary is None
    assert velour_datum[0].bbox == BoundingBox(16, 130, 70, 150)

    assert len(velour_datum[1].labels) == 1
    assert velour_datum[1].labels[0].tuple() == ("class_label", "person")
    assert velour_datum[1].image.uid == "img1"
    assert velour_datum[1].image.height == -1
    assert velour_datum[1].image.width == -1
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
    assert velour_datum.image.height == -1
    assert velour_datum.image.width == -1
    assert velour_datum.image.frame is None
    assert velour_datum.boundary is None
    assert velour_datum.bbox == BoundingBox(500, 220, 530, 260)

    # Item 3 - No object of interest
    velour_datum = chariot_parse_object_detection_annotation(
        chariot_dataset[2]
    )
    assert len(velour_datum) == 0


if __name__ == "__main__":

    from chariot.client import connect
    from chariot.datasets.dataset import get_datasets_in_project

    connect(host="https://production.chariot.striveworks.us")

    # List available datasets in project
    project_name = "Global"
    # project_name = "OnBoarding"
    dataset_list = get_datasets_in_project(
        limit=25, offset=0, project_name=project_name
    )

    lookup = {}
    print("Datasets")
    for i in range(len(dataset_list)):
        lookup[str(dataset_list[i].name).strip()] = dataset_list[i]
        print(" " + str(i) + ": " + dataset_list[i].name)

    chariot_ds = lookup["CIFAR-10"]
    # chariot_ds = lookup["Testing"]

    velour_client = Client(host="http://localhost:8000")
    # velour_client = Client(
    #     "https://velour.striveworks.us/api/",
    #     access_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik9rLURVZlRlcEczTlpuUjRCaFNhVCJ9.eyJlbWFpbCI6ImMuemFsb29tQHN0cml2ZXdvcmtzLnVzIiwiaXNzIjoiaHR0cHM6Ly92ZWxvdXIudXMuYXV0aDAuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTEzNjkyOTUwNjgyMzgxMzkzOTQwIiwiYXVkIjpbImh0dHBzOi8vdmVsb3VyLnN0cml2ZXdvcmtzLnVzLyIsImh0dHBzOi8vdmVsb3VyLnVzLmF1dGgwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2ODM3MjgzMzgsImV4cCI6MTY4MzgxNDczOCwiYXpwIjoiSkhzTDNXZ0N1ZVd5S2kwbW5CbDhvNDdyMFV4NlhHOFAiLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIn0.0d9dlO6NQETwvTdtFU-pPXFNrUDggqJAi09nhpf64b9TTDnrrhuKrcIfsK5OXTfJIt0lKxWx_Wbl_tfzYeeIGGxiImR07Qm_gl6_oGeKyJkLVYdgWqzoLRVbIMyRrpgs9YUdVHVXu9z-LHCQ4T0dgu0PETqmytoWrWSgO63xCgaAM8t6sGcHwvqYaL6bWSla4VQBdKkfGMTHUypsFopliES8xpTKp6SC8NXf9Ld2RMgOplLSO7hJJmyRXt5YNgPtZvBJZfWV3uLc93rQs55RSgUPW0W0a1qio7CI8YGYx1N0UX3FZFnIAxG85u3XTu_j6KYjiwWnFRbyGoUpsM05RQ",
    # )

    try:
        velour_client.delete_dataset(chariot_ds.name)
    except ClientException as err:
        print(err)

    velour_ds = chariot_ds_to_velour_ds(
        velour_client=velour_client, chariot_dataset=chariot_ds
    )

    print(velour_ds.get_labels())

    velour_client.delete_dataset(velour_ds.name)
