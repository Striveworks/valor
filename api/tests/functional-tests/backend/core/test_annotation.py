import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    return dataset_name


@pytest.fixture
def created_model(db: Session, model_name: str) -> str:
    model = schemas.Model(name=model_name)
    core.create_model(db, model=model)
    return model_name


@pytest.fixture
def datums(created_dataset) -> list[schemas.Datum]:
    return [
        schemas.Datum(
            uid=f"uid_{i}",
            dataset_name=created_dataset,
        )
        for i in range(3)
    ]


@pytest.fixture
def empty_groundtruths(datums) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(datum=datum, annotations=[]) for datum in datums
    ]


@pytest.fixture
def empty_predictions(created_model, datums) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model_name=created_model, datum=datum, annotations=[]
        )
        for datum in datums
    ]


def test_create_empty_annotations(
    db: Session,
    empty_groundtruths: list[schemas.GroundTruth],
    empty_predictions: list[schemas.Prediction],
    created_dataset,
):
    for gt in empty_groundtruths:
        core.create_groundtruth(db, gt)

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Annotation)
            .where(models.Annotation.task_type == enums.TaskType.EMPTY.value)
        )
        == 3
    )

    for pd in empty_predictions:
        core.create_prediction(db, pd)

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Annotation)
            .where(models.Annotation.task_type == enums.TaskType.EMPTY.value)
        )
        == 6
    )


def test_create_annotation_already_exists_error(
    db: Session,
    empty_groundtruths: list[schemas.GroundTruth],
    empty_predictions: list[schemas.Prediction],
):
    for gt in empty_groundtruths:
        core.create_groundtruth(db, gt)
    for pd in empty_predictions:
        core.create_prediction(db, pd)
    with pytest.raises(exceptions.DatumAlreadyExistsError):
        core.create_groundtruth(db, empty_groundtruths[0])
    with pytest.raises(exceptions.AnnotationAlreadyExistsError):
        core.create_prediction(db, empty_predictions[0])


def test_create_annotation_with_embedding(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    gt = schemas.GroundTruth(
        datum=schemas.Datum(uid="uid123", dataset_name=created_dataset),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
                labels=[schemas.Label(key="class", value="dog")],
            ),
        ],
    )

    pd = schemas.Prediction(
        model_name=created_model,
        datum=schemas.Datum(uid="uid123", dataset_name=created_dataset),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.EMBEDDING,
                embedding=[0.5, 0.5, 0.5],
            ),
        ],
    )

    core.create_groundtruth(db, gt)
    core.create_prediction(db, pd)

    assert (
        db.query(
            select(func.count()).select_from(models.Annotation).subquery()
        ).scalar()
        == 2
    )
    annotation = db.query(
        select(models.Annotation)
        .where(models.Annotation.model_id.isnot(None))
        .subquery()
    ).one_or_none()
    assert annotation.embedding_id is not None


def _convert_raster_to_polygon():
    from sqlalchemy import literal_column, type_coerce

    from valor_api.backend.core.geometry import GeometricValueType

    pixels_subquery = select(
        type_coerce(
            func.ST_PixelAsPoints(models.Annotation.raster, 1),
            type_=GeometricValueType,
        ).geom.label("geom")
    ).lateral("pixels")

    subquery = (
        select(
            models.Annotation.id.label("id"),
            func.ST_ConvexHull(func.ST_Collect(pixels_subquery.c.geom)).label(
                "raster_polygon"
            ),
        )
        .select_from(models.Annotation)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(
            pixels_subquery,
            literal_column(
                "true"
            ),  # Joining the lateral subquery doesn't require a condition
        )
        .group_by(models.Annotation.id)
        .subquery()
    )

    return subquery
