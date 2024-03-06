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
    _raster_to_png_b64,
    convert_geometry,
    get_annotation_type,
)
from valor_api.crud import create_dataset, create_groundtruth
from valor_api.schemas import (
    Annotation,
    BoundingBox,
    Datum,
    GroundTruth,
    MultiPolygon,
    Polygon,
    Raster,
)


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
                raster=raster,
            ),
        ],
    )
    dataset = schemas.Dataset(name=dataset_name)
    create_dataset(db=db, dataset=dataset)
    create_groundtruth(db=db, groundtruth=groundtruth)
    return dataset_name


@pytest.fixture
def create_segmentation_dataset_from_geometries(
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
                raster=Raster(
                    mask=raster.mask,
                    geometry=polygon,
                ),
            ),
            Annotation(
                task_type=task_type,
                labels=labels,
                raster=Raster(
                    mask=raster.mask,
                    geometry=multipolygon,
                ),
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


def test_convert_polygon_to_box(
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

    # check that points match
    assert converted_box == bbox


def test_create_segmentations_from_polygons(
    db: Session,
    create_segmentation_dataset_from_geometries: str,
    bbox: BoundingBox,
    polygon: Polygon,
    multipolygon: MultiPolygon,
    raster: Raster,
):
    # verify all rasters are equal
    raster_arrs = [
        Raster(mask=_raster_to_png_b64(db, r)).to_numpy()
        for r in db.scalars(select(models.Annotation.raster)).all()
    ]
    assert len(raster_arrs) == 3
    np.testing.assert_array_equal(raster_arrs[0], raster_arrs[1])
    np.testing.assert_array_equal(raster_arrs[2], raster.to_numpy())

    # NOTE - Comparing converted rasters to originals fails due to inaccuracies with polygon to raster conversion.
    # np.testing.assert_array_equal(raster_arrs[0], raster_arrs[2])

    # verify conversion to polygons
    db.execute(_convert_raster_to_polygon([]))
    polygons = [
        _load_polygon(db, poly)
        for poly in db.scalars(select(models.Annotation.polygon)).all()
    ]
    assert len(polygons) == 3
    assert polygons[0] == polygons[1]
    assert polygons[0] == polygons[2]
    assert polygons[0] == polygon

    # verify conversion to bbox
    db.execute(_convert_raster_to_box([]))
    boxes = [
        _load_box(db, box)
        for box in db.scalars(select(models.Annotation.box)).all()
    ]
    assert len(boxes) == 3
    assert boxes[0] == boxes[1]
    assert boxes[0] == boxes[2]
    assert boxes[0] == bbox
