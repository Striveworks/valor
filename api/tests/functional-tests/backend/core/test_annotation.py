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
