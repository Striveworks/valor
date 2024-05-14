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
def datums() -> list[schemas.Datum]:
    return [schemas.Datum(uid=f"uid_{i}") for i in range(3)]


@pytest.fixture
def empty_groundtruths(
    created_dataset: str, datums: list[schemas.Datum]
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset_name=created_dataset, datum=datum, annotations=[]
        )
        for datum in datums
    ]


@pytest.fixture
def empty_predictions(
    created_dataset: str, created_model: str, datums: list[schemas.Datum]
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            dataset_name=created_dataset,
            model_name=created_model,
            datum=datum,
            annotations=[],
        )
        for datum in datums
    ]


def test_create_empty_annotations(
    db: Session,
    empty_groundtruths: list[schemas.GroundTruth],
    empty_predictions: list[schemas.Prediction],
    created_dataset: str,
):
    core.create_groundtruths(db, empty_groundtruths)

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Annotation)
            .where(models.Annotation.task_type == enums.TaskType.EMPTY.value)
        )
        == 3
    )

    core.create_predictions(db, empty_predictions)

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

    core.create_groundtruths(db, empty_groundtruths)
    core.create_predictions(db, empty_predictions)
    with pytest.raises(exceptions.DatumsAlreadyExistsError):
        core.create_groundtruths(db, empty_groundtruths[0:1])
    with pytest.raises(exceptions.AnnotationAlreadyExistsError):
        core.create_predictions(db, empty_predictions[0:1])


def test_create_annotation_with_embedding(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    gt = schemas.GroundTruth(
        dataset_name=created_dataset,
        datum=schemas.Datum(uid="uid123"),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
                labels=[schemas.Label(key="class", value="dog")],
            ),
        ],
    )

    pd = schemas.Prediction(
        dataset_name=created_dataset,
        model_name=created_model,
        datum=schemas.Datum(uid="uid123"),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.EMBEDDING,
                embedding=[0.5, 0.5, 0.5],
            ),
        ],
    )

    core.create_groundtruths(db, [gt])
    core.create_predictions(db, [pd])

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
    assert annotation is not None
    assert annotation.embedding_id is not None
