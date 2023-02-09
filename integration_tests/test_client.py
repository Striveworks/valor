import json
from typing import Union

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from velour.client import Client, ClientException
from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    Image,
    Label,
    Point,
    PredictedDetection,
)

from velour_api import models, ops


def _list_of_points_from_wkt_polygon(
    db: Session, det: Union[GroundTruthDetection, PredictedDetection]
) -> list[tuple[int, int]]:
    geo = json.loads(db.scalar(det.boundary.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [tuple(p) for p in geo["coordinates"][0]]


def area(rect: BoundingPolygon) -> float:
    """Computes the area of a rectangle"""
    assert len(rect.points) == 4
    xs = [pt.x for pt in rect.points]
    ys = [pt.y for pt in rect.points]

    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1.points) == len(rect2.points) == 4

    xs1 = [pt.x for pt in rect1.points]
    xs2 = [pt.x for pt in rect2.points]

    ys1 = [pt.y for pt in rect1.points]
    ys2 = [pt.y for pt in rect2.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


@pytest.fixture
def db(client: Client):
    if len(client.get_datasets()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing datasets."
        )

    if len(client.get_all_labels()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing labels."
        )

    engine = create_engine("postgresql://postgres:password@localhost/postgres")
    sess = Session(engine)

    yield sess

    # cleanup by deleting all datasets and labels
    for dataset in client.get_datasets():
        client.delete_dataset(name=dataset["name"])

    labels = sess.scalars(select(models.Label))
    for label in labels:
        sess.delete(label)

    sess.commit()


@pytest.fixture
def rect1():
    return BoundingPolygon(
        [
            Point(x=10, y=10),
            Point(x=10, y=40),
            Point(x=60, y=40),
            Point(x=60, y=10),
        ]
    )


@pytest.fixture
def rect2():
    return BoundingPolygon(
        [
            Point(x=15, y=0),
            Point(x=70, y=0),
            Point(x=70, y=20),
            Point(x=15, y=20),
        ]
    )


@pytest.fixture
def rect3():
    return BoundingPolygon(
        [
            Point(x=158, y=10),
            Point(x=87, y=10),
            Point(x=87, y=820),
            Point(x=158, y=820),
        ]
    )


def test_create_dataset_with_detections(
    client: Client,
    rect1: BoundingPolygon,
    rect2: BoundingPolygon,
    rect3: BoundingPolygon,
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    """This test does the following
    - Creates a dataset
    - Adds groundtruth data to it in two batches
    - Verifies the images and labels have actually been added
    - Finalizes dataset
    - Tries to add more data and verifies an error is thrown
    """
    dset_name = "test dataset"
    dataset = client.create_dataset(dset_name)

    with pytest.raises(ClientException) as exc_info:
        client.create_dataset(dset_name)
    assert "already exists" in str(exc_info)

    dataset.add_groundtruth_detections(
        [
            GroundTruthDetection(
                boundary=rect1,
                labels=[Label(key="k1", value="v1")],
                image=Image(uri="uri1"),
            ),
            GroundTruthDetection(
                boundary=rect2,
                labels=[Label(key="k1", value="v1")],
                image=Image(uri="uri2"),
            ),
        ]
    )

    dataset.add_groundtruth_detections(
        [
            GroundTruthDetection(
                boundary=rect3,
                labels=[Label(key="k2", value="v2")],
                image=Image(uri="uri1"),
            )
        ]
    )

    # check that the dataset has two images
    images = dataset.get_images()
    assert len(images) == 2
    assert set([image.uri for image in images]) == {"uri1", "uri2"}

    # check that there are two labels
    labels = dataset.get_labels()
    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == {
        ("k1", "v1"),
        ("k2", "v2"),
    }

    dataset.finalize()
    # check that we get an error when trying to add more images
    # to the dataset since it is finalized
    with pytest.raises(ClientException) as exc_info:
        dataset.add_groundtruth_detections(
            [
                GroundTruthDetection(
                    boundary=rect3,
                    labels=[Label(key="k3", value="v3")],
                    image=Image(uri="uri8"),
                )
            ]
        )
    assert "since it is finalized" in str(exc_info)


def test_get_dataset(client: Client):
    pass


def test_boundary(client: Client, db: Session, rect1: BoundingPolygon):
    """Test consistency of boundary in backend and client"""
    dset_name = "test dataset"
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth_detections(
        [
            GroundTruthDetection(
                boundary=rect1,
                labels=[Label(key="k1", value="v1")],
                image=Image(uri="uri1"),
            )
        ]
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.GroundTruthDetection))

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect1.points])


def test_upload_predicted_detections(
    client: Client, session: Session, rect2: BoundingPolygon
):
    """Test that upload of a predicted detection from velour client to backend works"""
    pred_det = PredictedDetection(
        boundary=rect2, class_label="class-2", score=0.7
    )
    det_id = client.upload_predicted_detections([pred_det])[0]
    db_det = session.query(PredictedDetection).get(det_id)

    # check score
    assert db_det.score == pred_det.score == 0.7

    # check label
    assert db_det.class_label == pred_det.class_label

    # check boundary
    points = _list_of_points_from_wkt_polygon(session, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect2.points])


def test_iou(
    client: Client,
    db: Session,
    rect1: BoundingPolygon,
    rect2: BoundingPolygon,
):
    dset_name = "test dataset"
    client.create_dataset(dset_name)

    gt_det = GroundTruthDetection(
        boundary=rect1,
        class_label="class-1",
    )
    client.upload_groundtruth_detections([gt_det])[0]
    db_gt = db.scalar(select(GroundTruthDetection))

    pred_det = PredictedDetection(
        boundary=rect2, class_label="class-1", score=0.6
    )
    client.upload_predicted_detections([pred_det])[0]
    db_pred = db.scalar(select(PredictedDetection))

    assert ops.iou(db, db_gt, db_pred) == iou(rect1, rect2)
