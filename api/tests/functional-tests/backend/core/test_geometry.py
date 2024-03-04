import json

import numpy as np
import pytest
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend import models
from valor_api.backend.core import fetch_dataset
from valor_api.backend.core.geometry import (
    _convert_polygon_to_box,
    _convert_raster_to_box,
    _convert_raster_to_polygon,
    convert_geometry,
    get_annotation_type,
)
from valor_api.crud import create_dataset, create_groundtruth
from valor_api.schemas import (
    Annotation,
    BasicPolygon,
    BoundingBox,
    Datum,
    GroundTruth,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)


@pytest.fixture
def rotated_box_points() -> list[Point]:
    return [
        Point(x=4, y=0),
        Point(x=1, y=3),
        Point(x=4, y=6),
        Point(x=7, y=3),
    ]


@pytest.fixture
def bbox(rotated_box_points) -> BoundingBox:
    """Defined as the envelope of `rotated_box_points`."""
    minX = min([pt.x for pt in rotated_box_points])
    maxX = max([pt.x for pt in rotated_box_points])
    minY = min([pt.y for pt in rotated_box_points])
    maxY = max([pt.y for pt in rotated_box_points])
    return BoundingBox(
        polygon=BasicPolygon(
            points=[
                Point(x=minX, y=maxY),
                Point(x=maxX, y=maxY),
                Point(x=maxX, y=minY),
                Point(x=minX, y=minY),
            ]
        )
    )


@pytest.fixture
def polygon(rotated_box_points) -> Polygon:
    return Polygon(
        boundary=schemas.BasicPolygon(
            points=rotated_box_points,
        )
    )


@pytest.fixture
def multipolygon(polygon) -> MultiPolygon:
    return MultiPolygon(polygons=[polygon])


@pytest.fixture
def raster() -> Raster:
    """Rasterization of `rotated_box_points`."""
    r = np.zeros((10, 10))
    for y in range(0, 3):
        for x in range(4 - y, y + 5, 1):
            r[y, x] = 1
    for y in range(3, 8):
        for x in range(y - 2, 11 - y):
            r[y, x] = 1
    return Raster.from_numpy(r == 1)


@pytest.fixture
def create_classification_dataset(db: Session, dataset_name: str):
    create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )


@pytest.fixture
def create_object_detection_dataset(
    db: Session,
    dataset_name: str,
    bbox: BoundingBox,
    polygon: Polygon,
    multipolygon: MultiPolygon,
    raster: Raster,
):
    datum = Datum(
        uid="uid1",
        dataset_name=dataset_name,
    )
    task_type = enums.TaskType.OBJECT_DETECTION
    labels = [schemas.Label(key="k1", value="v1")]
    groundtruth = GroundTruth(
        datum=datum,
        annotations=[
            Annotation(
                task_type=task_type,
                labels=labels,
                bounding_box=bbox,
            ),
            Annotation(
                task_type=task_type,
                labels=labels,
                polygon=polygon,
            ),
            Annotation(
                task_type=task_type,
                labels=labels,
                multipolygon=multipolygon,
            ),
            Annotation(
                task_type=task_type,
                labels=labels,
                raster=raster,
            ),
        ],
    )
    dataset = schemas.Dataset(name=dataset_name)
    create_dataset(db=db, dataset=dataset)
    create_groundtruth(db=db, groundtruth=groundtruth)
    return dataset_name


def test_get_annotation_type(
    db: Session, dataset_name: str, create_classification_dataset
):
    # tests uncovered case where `AnnotationType.NONE` is returned.
    dataset = fetch_dataset(db, dataset_name)
    assert (
        get_annotation_type(db, enums.TaskType.CLASSIFICATION, dataset)
        == enums.AnnotationType.NONE
    )


def test_convert_geometry_input(
    db: Session, dataset_name: str, dataset_model_create
):
    dataset = fetch_dataset(db, dataset_name)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.NONE,
            target_type=enums.AnnotationType.BOX,
            dataset=None,
            model=None,
        )
    assert "source" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.NONE,
            dataset=None,
            model=None,
        )
    assert "target" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.RASTER,
            dataset=None,
            model=None,
        )
    assert "not capable of being converted" in str(e)

    with pytest.raises(ValueError):
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.BOX,
            dataset=dataset,
            model=None,
        )

    with pytest.raises(ValueError):
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.POLYGON,
            dataset=dataset,
            model=None,
        )


def _load_polygon(db: Session, polygon) -> Polygon:
    geom = json.loads(db.scalar(func.ST_AsGeoJSON(polygon)))
    return schemas.metadata.geojson_from_dict(data=geom).geometry()


def _load_box(db: Session, box) -> BoundingBox:
    return schemas.BoundingBox(polygon=_load_polygon(db, box).boundary)


def test_convert_from_raster(
    db: Session,
    create_object_detection_dataset: str,
    bbox: BoundingBox,
    polygon: Polygon,
):
    annotation_id = db.scalar(
        select(models.Annotation.id).where(
            and_(
                models.Annotation.box.is_(None),
                models.Annotation.polygon.is_(None),
                models.Annotation.raster.isnot(None),
            )
        )
    )
    assert annotation_id is not None

    q = _convert_raster_to_box([])
    db.execute(q)

    q = _convert_raster_to_polygon([])
    db.execute(q)

    annotation = db.query(
        select(models.Annotation)
        .where(models.Annotation.id == annotation_id)
        .subquery()
    ).one_or_none()
    assert annotation is not None

    assert annotation.box is not None
    assert annotation.polygon is not None
    assert annotation.raster is not None

    converted_box = _load_box(db, annotation.box)
    converted_polygon = _load_polygon(db, annotation.polygon)

    # check that points match
    assert converted_box == bbox
    assert converted_polygon == polygon


def test_convert_polygon_to_bbox(
    db: Session,
    create_object_detection_dataset: str,
    bbox: BoundingBox,
):
    annotation_id = db.scalar(
        select(models.Annotation.id).where(
            and_(
                models.Annotation.box.is_(None),
                models.Annotation.polygon.isnot(None),
                models.Annotation.raster.is_(None),
            )
        )
    )
    assert annotation_id is not None

    q = _convert_polygon_to_box([])
    db.execute(q)

    annotation = db.query(
        select(models.Annotation)
        .where(models.Annotation.id == annotation_id)
        .subquery()
    ).one_or_none()
    assert annotation is not None

    assert annotation.box is not None
    assert annotation.polygon is not None
    assert annotation.raster is None

    converted_box = _load_box(db, annotation.box)

    # rotate points
    converted_box.polygon.points = converted_box.polygon.points[1:] + [
        converted_box.polygon.points[0]
    ]

    # check that points match
    assert converted_box == bbox
