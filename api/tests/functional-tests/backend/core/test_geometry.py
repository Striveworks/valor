import json

import numpy as np
import pytest
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend import models
from valor_api.backend.core import fetch_dataset
from valor_api.backend.core.geometry import (
    _convert_multipolygon_to_box,
    _convert_multipolygon_to_polygon,
    _convert_polygon_to_box,
    _convert_raster_to_box,
    _convert_raster_to_multipolygon,
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
    Prediction,
    Raster,
)


@pytest.fixture
def rotated_box_points() -> list[Point]:
    return [
        Point(x=1, y=3),
        Point(x=4, y=6),
        Point(x=7, y=3),
        Point(x=4, y=0),
        Point(x=1, y=3),
    ]


@pytest.fixture
def bbox() -> BoundingBox:
    """Defined as the envelope of `rotated_box_points`."""
    return BoundingBox.from_extrema(
        xmin=1,
        xmax=7,
        ymin=0,
        ymax=6,
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
    for y in range(-1, -5, -1):
        for x in range(5 + y, 4 - y, 1):
            r[y, x] = 1
    for y in range(-5, -8, -1):

        # 2 - 6   -5       -3-y           12 + y
        # 3 - 5   -6       -3-y           12 + y
        # 4 -     -7       -3-y           12 + y

        for x in range(-3 - y, 12 + y):
            print(y, x)
            r[y, x] = 1
    print()
    print(r)
    return Raster.from_numpy(r == 1)


@pytest.fixture
def create_clf_dataset(db: Session, dataset_name: str):
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
def create_objdet_dataset(
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
    db: Session, dataset_name: str, create_clf_dataset
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
    assert "Source type" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.NONE,
            dataset=None,
            model=None,
        )
    assert "Target type" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.RASTER,
            dataset=None,
            model=None,
        )
    assert "not capable of being converted" in str(e)

    with pytest.raises(NotImplementedError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.BOX,
            dataset=dataset,
            model=None,
        )
    assert "currently unsupported" in str(e)

    with pytest.raises(NotImplementedError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.POLYGON,
            dataset=dataset,
            model=None,
        )
    assert "currently unsupported" in str(e)


def test__convert_raster_to_bbox(
    db: Session,
    create_objdet_dataset: str,
    bbox: BoundingBox,
):
    dataset = fetch_dataset(db=db, name=create_objdet_dataset)

    annotation_id = db.scalar(
        select(models.Annotation.id).where(
            and_(
                models.Annotation.box.is_(None),
                models.Annotation.polygon.is_(None),
                models.Annotation.multipolygon.is_(None),
                models.Annotation.raster.isnot(None),
            )
        )
    )
    assert annotation_id is not None

    q = _convert_raster_to_box(dataset_id=dataset.id)
    db.execute(q)

    annotation = db.query(
        select(models.Annotation)
        .where(models.Annotation.id == annotation_id)
        .subquery()
    ).one_or_none()
    assert annotation is not None

    assert annotation.box is not None
    assert annotation.polygon is None
    assert annotation.multipolygon is None
    assert annotation.raster is not None

    geom = json.loads(
        db.scalar(func.ST_AsGeoJSON(func.ST_Envelope(annotation.raster)))
    )
    converted_box = schemas.BoundingBox(
        polygon=schemas.metadata.geojson_from_dict(data=geom)
        .geometry()
        .boundary,  # type: ignore - this is guaranteed to be a polygon
    )

    # check that points match
    for original_point in bbox.polygon.points:
        assert original_point in converted_box.polygon.points


def test_convert_raster_to_polygon():
    pass


def test_convert_polygon_to_bbox():
    pass
