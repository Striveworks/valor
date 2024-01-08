import pytest

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from velour_api import schemas, exceptions, enums
from velour_api.backend import core, models


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
def created_models(db: Session) -> list[str]:
    model1 = schemas.Model(name="model1")
    model2 = schemas.Model(name="model2")
    core.create_model(db, model=model1)
    core.create_model(db, model=model2)
    return ["model1", "model2"]


def test_created_model(db: Session, created_model):
    model = db.query(
        select(models.Model)
        .where(models.Model.name == created_model)
        .subquery()
    ).one_or_none()
    assert model is not None
    assert model.name == created_model
    assert model.meta == {}


def test_fetch_model(db: Session, created_model):
    model = core.fetch_model(db, created_model)
    assert model is not None
    assert model.name == created_model
    assert model.meta == {}

    with pytest.raises(exceptions.ModelDoesNotExistError):
        core.fetch_model(db, "some_nonexistent_model")


def test_get_model(db: Session, created_model):
    model = core.get_model(db, created_model)
    assert model is not None
    assert model.name == created_model
    assert model.metadata == {}
    assert model.geospatial is None

    with pytest.raises(exceptions.ModelDoesNotExistError):
        core.get_model(db, "some_nonexistent_model")


def test_get_models(db: Session, created_models):
    models = core.get_models(db)
    for model in models:
        assert model.name in created_models


def test_model_status(db: Session, created_model, created_dataset):
    # creating
    assert core.get_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model
    ) == enums.TableStatus.CREATING

    # attempt to finalize before dataset
    with pytest.raises(exceptions.DatasetNotFinalizedError):
        core.set_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
            status=enums.TableStatus.FINALIZED,
        )
    assert core.get_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model
    ) == enums.TableStatus.CREATING

    # finalize dataset
    core.set_dataset_status(
        db=db,
        name=created_dataset,
        status=enums.TableStatus.FINALIZED,
    )
    core.set_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model,
        status=enums.TableStatus.FINALIZED,
    )
    assert core.get_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model
    ) == enums.TableStatus.FINALIZED

    # test others
    core.set_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model,
        status=enums.TableStatus.FINALIZED,
    )
    with pytest.raises(exceptions.ModelStateError):
        core.set_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
            status=enums.TableStatus.CREATING,
        )

    # deleting
    core.set_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model,
        status=enums.TableStatus.DELETING,
    )
    assert core.get_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model
    ) == enums.TableStatus.DELETING

    # test others
    with pytest.raises(exceptions.ModelStateError):
        core.set_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
            status=enums.TableStatus.CREATING,
        )
    with pytest.raises(exceptions.ModelStateError):
        core.set_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
            status=enums.TableStatus.FINALIZED,
        )


def test_delete_model(db: Session):
    core.create_model(
        db=db,
        model=schemas.Model(name="model1")
    )

    assert db.scalar(
        select(func.count())
        .select_from(models.Model)
        .where(models.Model.name =="model1")
    ) == 1

    core.delete_model(
        db=db,
        name="model1"
    )

    assert db.scalar(
        select(func.count())
        .select_from(models.Model)
        .where(models.Model.name =="model1")
    ) == 0

