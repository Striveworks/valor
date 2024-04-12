import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


@pytest.fixture
def created_models(db: Session) -> list[str]:
    models = []
    for i in range(10):
        model = schemas.Model(name=f"model{i}")
        core.create_model(db, model=model)
        models.append(f"model{i}")

    return models


def test_create_model(db: Session, created_model):
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

    with pytest.raises(exceptions.ModelDoesNotExistError):
        core.get_model(db, "some_nonexistent_model")


def test_get_models(db: Session, created_models):
    models, headers = core.get_models(db)
    for model in models:
        assert model.name in created_models
    assert headers == {"content-range": "items 0-9/10"}

    # test pagination
    with pytest.raises(ValueError):
        # offset is greater than the number of items returned in query
        models, headers = core.get_models(db, offset=100, limit=2)

    models, headers = core.get_models(db, offset=5, limit=2)
    assert [model.name for model in models] == ["model4", "model3"]
    assert headers == {"content-range": "items 5-6/10"}

    models, headers = core.get_models(db, offset=2, limit=7)
    assert [model.name for model in models] == [
        f"model{i}" for i in range(7, 0, -1)
    ]
    assert headers == {"content-range": "items 2-8/10"}


def test_model_status(db: Session, created_model, created_dataset):
    # creating
    assert (
        core.get_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
        )
        == enums.TableStatus.CREATING
    )

    # attempt to finalize before dataset
    with pytest.raises(exceptions.DatasetNotFinalizedError):
        core.set_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
            status=enums.TableStatus.FINALIZED,
        )
    assert (
        core.get_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
        )
        == enums.TableStatus.CREATING
    )

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
    assert (
        core.get_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
        )
        == enums.TableStatus.FINALIZED
    )

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
    assert (
        core.get_model_status(
            db=db,
            dataset_name=created_dataset,
            model_name=created_model,
        )
        == enums.TableStatus.DELETING
    )

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


def test_model_status_with_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create an evaluation
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    created, _ = core.create_or_get_evaluations(
        db,
        schemas.EvaluationRequest(
            model_names=[created_model],
            datum_filter=schemas.Filter(dataset_names=[created_dataset]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        ),
    )
    assert len(created) == 1
    evaluation_id = created[0].id

    # set the evaluation to the running state
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.RUNNING
    )

    # test that deletion is blocked while evaluation is running
    with pytest.raises(exceptions.EvaluationRunningError):
        core.set_model_status(
            db,
            created_dataset,
            created_model,
            enums.TableStatus.DELETING,
        )

    # set the evaluation to the done state
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # test that deletion is unblocked when evaluation is DONE
    core.set_model_status(
        db,
        created_dataset,
        created_model,
        enums.TableStatus.DELETING,
    )


def test_delete_model(db: Session):
    core.create_model(db=db, model=schemas.Model(name="model1"))

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Model)
            .where(models.Model.name == "model1")
        )
        == 1
    )

    core.delete_model(db=db, name="model1")

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Model)
            .where(models.Model.name == "model1")
        )
        == 0
    )
