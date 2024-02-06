import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    core.create_groundtruth(
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
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid3", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    return dataset_name


@pytest.fixture
def created_model(db: Session, model_name: str, created_dataset: str) -> str:
    model = schemas.Model(name=model_name)
    core.create_model(db, model=model)
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid1", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid2", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid3", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    return model_name


@pytest.fixture
def created_models(db: Session) -> list[str]:
    model1 = schemas.Model(name="model1")
    model2 = schemas.Model(name="model2")
    core.create_model(db, model=model1)
    core.create_model(db, model=model2)
    return ["model1", "model2"]


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
    assert model.geospatial is None

    with pytest.raises(exceptions.ModelDoesNotExistError):
        core.get_model(db, "some_nonexistent_model")


def test_get_models(db: Session, created_models):
    models = core.get_models(db)
    for model in models:
        assert model.name in created_models


def test_model_status(db: Session, created_model, created_dataset):
    # creating
    assert (
        core.get_model_status(
            db=db, dataset_name=created_dataset, model_name=created_model
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
            db=db, dataset_name=created_dataset, model_name=created_model
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
            db=db, dataset_name=created_dataset, model_name=created_model
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
            db=db, dataset_name=created_dataset, model_name=created_model
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
            db, created_dataset, created_model, enums.TableStatus.DELETING
        )

    # set the evaluation to the done state
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # test that deletion is unblocked when evaluation is DONE
    core.set_model_status(
        db, created_dataset, created_model, enums.TableStatus.DELETING
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
