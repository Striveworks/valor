import numpy as np
import pytest
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend import models
from valor_api.backend.core import fetch_dataset, get_groundtruth
from valor_api.backend.core.geometry import (
    _convert_polygon_to_box,
    _convert_raster_to_box,
    _convert_raster_to_polygon,
    _raster_to_png_b64,
    convert_geometry,
)
from valor_api.crud import create_dataset, create_groundtruths
from valor_api.schemas import (
    Annotation,
    Box,
    Datum,
    GroundTruth,
    MultiPolygon,
    Polygon,
    Raster,
)


@pytest.fixture
def create_classification_dataset(db: Session, dataset_name: str):
    create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dataset_name,
                datum=schemas.Datum(uid="uid1"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="k1", value="v1")],
                    )
                ],
            )
        ],
    )


@pytest.fixture
def create_object_detection_dataset(
    db: Session,
    dataset_name: str,
    bbox: Box,
    polygon: Polygon,
    raster: Raster,
):
    datum = Datum(uid="uid1")
    labels = [schemas.Label(key="k1", value="v1")]
    groundtruth = GroundTruth(
        dataset_name=dataset_name,
        datum=datum,
        annotations=[
            Annotation(
                labels=labels,
                bounding_box=bbox,
                is_instance=True,
            ),
            Annotation(
                labels=labels,
                polygon=polygon,
                is_instance=True,
            ),
            Annotation(
                labels=labels,
                raster=raster,
                is_instance=True,
            ),
        ],
    )
    dataset = schemas.Dataset(name=dataset_name)
    create_dataset(db=db, dataset=dataset)
    create_groundtruths(db=db, groundtruths=[groundtruth])
    return dataset_name


@pytest.fixture
def create_segmentation_dataset_from_geometries(
    db: Session,
    dataset_name: str,
    polygon: Polygon,
    multipolygon: MultiPolygon,
    raster: Raster,
):
    datum = Datum(uid="uid1")
    labels = [schemas.Label(key="k1", value="v1")]
    groundtruth = GroundTruth(
        dataset_name=dataset_name,
        datum=datum,
        annotations=[
            Annotation(
                labels=labels,
                raster=Raster(
                    mask=raster.mask,
                    geometry=polygon,
                ),
                is_instance=True,
            ),
            Annotation(
                labels=labels,
                raster=Raster(
                    mask=raster.mask,
                    geometry=multipolygon,
                ),
                is_instance=True,
            ),
            Annotation(labels=labels, raster=raster, is_instance=True),
        ],
    )
    dataset = schemas.Dataset(name=dataset_name)
    create_dataset(db=db, dataset=dataset)
    create_groundtruths(db=db, groundtruths=[groundtruth])
    return dataset_name


def test_convert_geometry_input(
    db: Session, dataset_name: str, dataset_model_create
):
    dataset = fetch_dataset(db, dataset_name)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.NONE,
            target_type=enums.AnnotationType.BOX,
            dataset=None,  # type: ignore - purposefully throwing error
            model=None,
        )
    assert "source" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.NONE,
            dataset=None,  # type: ignore - purposefully throwing error
            model=None,
        )
    assert "target" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.RASTER,
            dataset=None,  # type: ignore - purposefully throwing error
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


def _load_polygon(db: Session, polygon: Polygon) -> Polygon:
    return Polygon.from_json(db.scalar(func.ST_AsGeoJSON(polygon)))


def _load_box(db: Session, box) -> Box:
    return schemas.Box(value=_load_polygon(db, box).value)


def test_convert_from_raster(
    db: Session,
    create_object_detection_dataset: str,
    bbox: Box,
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
    assert converted_box == Box(
        value=[
            [
                (1.0, 0.0),
                (1.0, 7.0),
                (8.0, 7.0),
                (8.0, 0.0),
                (1.0, 0.0),
            ]
        ]
    )
    assert converted_polygon == polygon


def test_convert_polygon_to_box(
    db: Session,
    create_object_detection_dataset: str,
    bbox: Box,
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


def test_create_raster_from_polygons(
    db: Session,
    create_segmentation_dataset_from_geometries: str,
    bbox: Box,
    polygon: Polygon,
    multipolygon: MultiPolygon,
    raster: Raster,
):
    # NOTE - Comparing converted rasters to originals fails due to inaccuracies with polygon to raster conversion.
    #         This is the raster that will be created through the conversion.
    converted_raster = Raster.from_numpy(
        np.array(
            [  # 0 1 2 3 4 5 6 7 8 9
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 0
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # 1
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # 2
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # 3
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # 4
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
            ]
        )
        == 1
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            converted_raster.to_numpy(), raster.to_numpy()
        )

    # verify all rasters are equal
    raster_arrs = [
        Raster(mask=_raster_to_png_b64(db, r)).to_numpy()
        for r in db.scalars(select(models.Annotation.raster)).all()
    ]
    assert len(raster_arrs) == 3

    np.testing.assert_array_equal(raster_arrs[0], raster_arrs[1])
    np.testing.assert_array_equal(
        raster_arrs[0], converted_raster.to_numpy()
    )  # converted rasters are equal

    np.testing.assert_array_equal(
        raster_arrs[2], raster.to_numpy()
    )  # directly ingested raster is the same
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            raster_arrs[0], raster_arrs[2]
        )  # ingested raster not equal to polygon-raster

    # NOTE - Conversion error causes this.
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(raster_arrs[0], raster_arrs[2])

    # verify no polygons or boxes exist
    assert (
        db.scalar(
            select(func.count(models.Annotation.id)).where(
                or_(
                    models.Annotation.box.isnot(None),
                    models.Annotation.polygon.isnot(None),
                )
            )
        )
        == 0
    )

    # verify conversion to polygons
    try:
        db.execute(_convert_raster_to_polygon([]))
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    polygons = [
        _load_polygon(db, poly)
        for poly in db.scalars(select(models.Annotation.polygon)).all()
    ]
    assert len(polygons) == 3

    # NOTE - Due to the issues in rasterization, converting back to polygon results in a new polygon.
    converted_polygon = Polygon(
        value=[
            [
                (4, 0),
                (2, 2),
                (2, 3),
                (4, 5),
                (6, 3),
                (6, 2),
                (4, 0),
            ]
        ]
    )
    assert polygons[0] == polygons[1]
    assert (
        polygons[0] == converted_polygon
    )  # corrupted raster converts to inccorect polygon
    assert (
        polygons[2] == polygon
    )  # uncorrupted raster converts to correct polygon


def test_create_raster_from_polygons_with_decimal_coordinates(
    db: Session,
    dataset_name: str,
    polygon: Polygon,
    raster: Raster,
):
    # alter polygon to be offset by 0.1
    assert polygon.to_wkt() == "POLYGON ((4 0, 1 3, 4 6, 7 3, 4 0))"
    polygon.value = [
        [(point[0] + 0.1, point[1] + 0.5) for point in subpolygon]
        for subpolygon in polygon.value
    ]
    assert (
        polygon.to_wkt()
        == "POLYGON ((4.1 0.5, 1.1 3.5, 4.1 6.5, 7.1 3.5, 4.1 0.5))"
    )

    # create raster annotation
    datum = Datum(uid="uid1")
    labels = [schemas.Label(key="k1", value="v1")]
    groundtruth = GroundTruth(
        dataset_name=dataset_name,
        datum=datum,
        annotations=[
            Annotation(
                labels=labels,
                raster=Raster(
                    mask=raster.mask,
                    geometry=polygon,
                ),
            ),
        ],
    )
    dataset = schemas.Dataset(name=dataset_name)
    create_dataset(db=db, dataset=dataset)
    create_groundtruths(db=db, groundtruths=[groundtruth])

    # retrieve the raster from the database to see if it has been converted.
    groundtruth = get_groundtruth(
        db=db, dataset_name=dataset_name, datum_uid="uid1"
    )
    assert groundtruth.annotations[0].raster
    # note that postgis rasterization is lossy
    arr = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        == 1
    )
    assert (groundtruth.annotations[0].raster.array == arr).all()

    # convert raster into a polygon
    polygon_subquery = (
        select(models.Annotation.polygon)
        .where(models.Annotation.polygon.isnot(None))
        .subquery()
    )
    assert len(db.query(polygon_subquery).all()) == 0
    q = _convert_raster_to_polygon([])
    db.execute(q)
    assert len(db.query(polygon_subquery).all()) == 1

    # check polygon in WKT format
    db_polygon = db.scalar(select(func.ST_AsText(models.Annotation.polygon)))
    assert db_polygon
    # note that postgis rasterization is lossy
    assert db_polygon == "POLYGON((3 1,1 3,3 5,4 5,6 3,4 1,3 1))"
