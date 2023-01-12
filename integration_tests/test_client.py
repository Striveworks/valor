import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from velour.client import Client
from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    Point,
    PredictedDetection,
)

from velour_api import ops
from velour_api.crud import _list_of_points_from_wkt_polygon
from velour_api.models import Detection


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


@pytest.fixture
def session():
    engine = create_engine("postgresql://postgres:password@localhost/postgres")
    return Session(engine)


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


def area(rect: BoundingPolygon) -> float:
    assert len(rect.points) == 4
    xs = [pt.x for pt in rect.points]
    ys = [pt.y for pt in rect.points]

    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    assert len(rect1.points) == len(rect2.points) == 4

    xs1 = [pt.x for pt in rect1.points]
    xs2 = [pt.x for pt in rect2.points]

    ys1 = [pt.y for pt in rect1.points]
    ys2 = [pt.y for pt in rect2.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    return (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)


def iou(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    return intersection_area(rect1, rect2) / (area(rect1) + area(rect2))


def test_upload_groundtruth_detection(
    client: Client, session: Session, rect1: BoundingPolygon
):
    """Test that upload of a groundtruth detection from velour client to backend works"""
    gt_det = GroundTruthDetection(
        boundary=rect1,
        class_label="class-1",
    )
    det_id = client.upload_groundtruth_detections([gt_det])[0]
    db_det = session.query(Detection).get(det_id)

    # check score is -1 since its a groundtruth detection
    assert db_det.score == -1

    # check label
    assert db_det.class_label == gt_det.class_label

    # check boundary
    points = _list_of_points_from_wkt_polygon(session, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect1.points])


def test_upload_predicted_detections(
    client: Client, session: Session, rect2: BoundingPolygon
):
    """Test that upload of a predicted detection from velour client to backend works"""
    pred_det = PredictedDetection(
        boundary=rect2, class_label="class-2", score=0.7
    )
    det_id = client.upload_predicted_detections([pred_det])[0]
    db_det = session.query(Detection).get(det_id)

    # check score
    assert db_det.score == pred_det.score == 0.7

    # check label
    assert db_det.class_label == pred_det.class_label

    # check boundary
    points = _list_of_points_from_wkt_polygon(session, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect2.points])


def test_iou(
    client: Client,
    session: Session,
    rect1: BoundingPolygon,
    rect2: BoundingPolygon,
):

    gt_det = GroundTruthDetection(
        boundary=rect1,
        class_label="class-1",
    )
    gt_id = client.upload_groundtruth_detections([gt_det])[0]
    db_gt = session.query(Detection).get(gt_id)

    pred_det = PredictedDetection(
        boundary=rect2, class_label="class-1", score=0.6
    )
    pred_id = client.upload_predicted_detections([pred_det])[0]
    db_pred = session.query(Detection).get(pred_id)

    assert ops.iou(session, db_gt, db_pred) == iou(rect1, rect2)
