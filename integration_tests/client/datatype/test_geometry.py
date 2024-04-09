""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import json

import numpy as np
from geoalchemy2.functions import ST_Area, ST_Intersection, ST_Union
from sqlalchemy import select
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import TaskType
from valor.metatypes import Datum
from valor.schemas import Box, Polygon, Raster
from valor_api.backend import models


def _generate_mask(
    height: int,
    width: int,
    minimum_mask_percent: float = 0.05,
    maximum_mask_percent: float = 0.4,
) -> np.ndarray:
    """Generate a random mask for an image with a given height and width"""
    mask_cutoff = np.random.uniform(minimum_mask_percent, maximum_mask_percent)
    mask = (np.random.random((height, width))) < mask_cutoff

    return mask


def _list_of_points_from_wkt_polygon(
    db: Session, det: models.Annotation
) -> list[tuple[float, float]]:
    geo = json.loads(db.scalar(det.polygon.ST_AsGeoJSON()) or "")
    assert len(geo["coordinates"]) == 1
    return [(p[0], p[1]) for p in geo["coordinates"][0]]


def area(rect: list[tuple[float, float]]) -> float:
    """Computes the area of a rectangle"""
    assert len(rect) == 5
    xs = [pt[0] for pt in rect]
    ys = [pt[1] for pt in rect]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(
    rect1: list[tuple[float, float]], rect2: list[tuple[float, float]]
) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1) == len(rect2) == 5

    xs1 = [pt[0] for pt in rect1]
    xs2 = [pt[0] for pt in rect2]

    ys1 = [pt[1] for pt in rect1]
    ys2 = [pt[1] for pt in rect2]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(
    rect1: list[tuple[float, float]], rect2: list[tuple[float, float]]
) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def test_boundary(
    db: Session,
    client: Client,
    dataset_name: str,
    rect1: list[tuple[float, float]],
    img1: Datum,
):
    """Test consistency of boundary in back end and client"""
    dataset = Dataset.create(dataset_name)
    rect1_poly = Polygon([rect1])
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=rect1_poly,
                )
            ],
        )
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.Annotation))
    assert db_det

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)
    assert points == rect1_poly.boundary


def test_iou(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    img1: Datum,
):
    rect1_poly = Polygon([rect1])
    rect2_poly = Polygon([rect2])

    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k", value="v")],
                    polygon=rect1_poly,
                )
            ],
        )
    )
    dataset.finalize()
    annotation = db.scalar(select(models.Annotation))
    assert annotation is not None
    db_gt = annotation.polygon

    model = Model.create(model_name)
    model.add_prediction(
        dataset,
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    polygon=rect2_poly,
                    labels=[Label(key="k", value="v", score=0.6)],
                )
            ],
        ),
    )
    model.finalize_inferences(dataset)
    annotation2 = db.scalar(
        select(models.Annotation).where(models.Annotation.model_id.isnot(None))
    )
    assert annotation2 is not None
    db_pred = annotation2.polygon

    # scraped from valor_api back end
    gintersection = ST_Intersection(db_gt, db_pred)
    gunion = ST_Union(db_gt, db_pred)
    iou_computation = ST_Area(gintersection) / ST_Area(gunion)

    assert iou(rect1_poly.boundary, rect2_poly.boundary) == db.scalar(
        select(iou_computation)
    )


def test_add_raster_and_boundary_box(
    client: Client,
    dataset_name: str,
    img1: Datum,
):
    img_size = [900, 300]
    mask = _generate_mask(height=img_size[0], width=img_size[1])
    raster = Raster.from_numpy(mask)

    gt = GroundTruth(
        datum=img1,
        annotations=[
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k3", value="v3")],
                bounding_box=Box.from_extrema(
                    xmin=10, ymin=10, xmax=60, ymax=40
                ),
                raster=raster,
            )
        ],
    )

    dataset = Dataset.create(dataset_name)

    dataset.add_groundtruth(gt)

    fetched_gt = dataset.get_groundtruth("uid1")

    assert fetched_gt
    assert (
        fetched_gt.annotations[0].raster is not None
    ), "Raster doesn't exist on fetched gt"
    assert (
        fetched_gt.annotations[0].bounding_box is not None
    ), "Bounding box doesn't exist on fetched gt"

    client.delete_dataset(dataset_name, timeout=30)
