""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
import json

from geoalchemy2.functions import ST_Area, ST_Intersection, ST_Union
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, GroundTruth, Label, Model, Prediction
from velour.client import Client
from velour.data_generation import _generate_mask
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox, Point, Polygon, Raster
from velour_api.backend import models


def bbox_to_poly(bbox: BoundingBox) -> Polygon:
    return Polygon(
        boundary=bbox.polygon,
        holes=None,
    )


def _list_of_points_from_wkt_polygon(
    db: Session, det: models.Annotation
) -> list[Point]:
    geo = json.loads(db.scalar(det.polygon.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [Point(p[0], p[1]) for p in geo["coordinates"][0][:-1]]


def area(rect: Polygon) -> float:
    """Computes the area of a rectangle"""
    assert len(rect.boundary.points) == 4
    xs = [pt.x for pt in rect.boundary.points]
    ys = [pt.y for pt in rect.boundary.points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: Polygon, rect2: Polygon) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1.boundary.points) == len(rect2.boundary.points) == 4

    xs1 = [pt.x for pt in rect1.boundary.points]
    xs2 = [pt.x for pt in rect2.boundary.points]

    ys1 = [pt.y for pt in rect1.boundary.points]
    ys2 = [pt.y for pt in rect2.boundary.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: Polygon, rect2: Polygon) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def test_boundary(
    db: Session,
    client: Client,
    dataset_name: str,
    rect1: Polygon,
    img1: ImageMetadata,
):
    """Test consistency of boundary in backend and client"""
    dataset = Dataset.create(client, dataset_name)
    rect1_poly = bbox_to_poly(rect1)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=rect1_poly,
                )
            ],
        )
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.Annotation))

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)

    assert set(points) == set([pt for pt in rect1_poly.boundary.points])


def test_iou(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    rect1: Polygon,
    rect2: Polygon,
    img1: ImageMetadata,
):
    rect1_poly = bbox_to_poly(rect1)
    rect2_poly = bbox_to_poly(rect2)

    dataset = Dataset.create(client, dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label("k", "v")],
                    polygon=rect1_poly,
                )
            ],
        )
    )
    dataset.finalize()
    db_gt = db.scalar(select(models.Annotation)).polygon

    model = Model.create(client, model_name)
    model.add_prediction(
        Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    polygon=rect2_poly,
                    labels=[Label("k", "v", score=0.6)],
                )
            ],
        )
    )
    model.finalize_inferences(dataset)
    db_pred = db.scalar(
        select(models.Annotation).where(models.Annotation.model_id.isnot(None))
    ).polygon

    # scraped from velour_api backend
    gintersection = ST_Intersection(db_gt, db_pred)
    gunion = ST_Union(db_gt, db_pred)
    iou_computation = ST_Area(gintersection) / ST_Area(gunion)

    assert iou(rect1_poly, rect2_poly) == db.scalar(select(iou_computation))


def test_add_raster_and_boundary_box(
    client: Client,
    dataset_name: str,
    img1: ImageMetadata,
):
    img_size = [900, 300]
    mask = _generate_mask(height=img_size[0], width=img_size[1])
    raster = Raster.from_numpy(mask)

    gt = GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.DETECTION,
                labels=[Label(key="k3", value="v3")],
                bounding_box=BoundingBox.from_extrema(
                    xmin=10, ymin=10, xmax=60, ymax=40
                ),
                raster=raster,
            )
        ],
    )

    dataset = Dataset.create(client, dataset_name)

    dataset.add_groundtruth(gt)

    fetched_gt = dataset.get_groundtruth("uid1")

    assert (
        fetched_gt.annotations[0].raster is not None
    ), "Raster doesn't exist on fetched gt"
    assert (
        fetched_gt.annotations[0].bounding_box is not None
    ), "Bounding box doesn't exist on fetched gt"

    client.delete_dataset(dataset_name, timeout=30)
