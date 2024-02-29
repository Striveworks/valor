import numpy as np
import pytest
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend.core import fetch_dataset
from valor_api.backend.core.geometry import (
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
        Point(x=0, y=3),
        Point(x=3, y=6),
        Point(x=6, y=3),
        Point(x=3, y=0),
        Point(x=0, y=3),
    ]


@pytest.fixture
def bbox() -> BoundingBox:
    """Defined as the envelope of `rotated_box_points`."""
    return BoundingBox.from_extrema(
        xmin=0,
        xmax=6,
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
    return MultiPolygon(
        polygons=[polygon]
    )


@pytest.fixture
def raster() -> Raster:
    """Rasterization of `rotated_box_points`."""
    r = np.zeros((10,10))
    for y in range(-1, -5, -1):
        for x in range(4+y, 3-y, 1):
            r[y,x] = 1
    for y in range(-5, -8, -1):
        for x in range(-y-4, 11+y, 1):
            print()
            print(y, x)
            r[y,x] = 1
    return r


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
    bbox_groundtruth = GroundTruth(
        datum=Datum(
            uid="uid1",
            dataset_name=dataset_name,
        ),
        annotations=[
            Annotation(
                task_type=enums.TaskType.OBJECT_DETECTION,
                labels=[schemas.Label(key="k1", value="v1")]
            )
        ]
    )


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


def test_convert_from_raster(raster, multipolygon, bbox, polygon):



def test_convert_raster_to_polygon():
    pass


def test_convert_polygon_to_bbox():
    pass
