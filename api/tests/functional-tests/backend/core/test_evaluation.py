import pytest
from sqlalchemy.orm import Session

from velour_api import enums, schemas, exceptions
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


def test_create_evaluation(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_id = core.create_evaluation(db, job_request_1)
    job_request_1.id == evaluation_id

    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.PENDING

    # test duplication check
    with pytest.raises(exceptions.EvaluationAlreadyExistsError):
        core.create_evaluation(db, job_request_1)

    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.PENDING

    rows = db.query(models.Evaluation).all()
    assert len(rows) == 1
    assert rows[0].id == evaluation_id
    assert rows[0].settings == job_request_1.settings.model_dump()


def test_fetch_evaluation_from_id(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_1 = core.create_evaluation(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.SEGMENTATION,
    )
    evaluation_2 = core.create_evaluation(db, job_request_2)

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_1)
    assert fetched_evaluation.id == evaluation_1
    assert fetched_evaluation.task_type == enums.TaskType.CLASSIFICATION

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_2)
    assert fetched_evaluation.id == evaluation_2
    assert fetched_evaluation.task_type == enums.TaskType.SEGMENTATION


def test_fetch_evaluation_from_job_request(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_1 = core.create_evaluation(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.SEGMENTATION,
    )
    evaluation_2 = core.create_evaluation(db, job_request_2)

    fetched_evaluation = core.fetch_evaluation_from_job_request(db, job_request_1)
    assert fetched_evaluation.id == evaluation_1
    assert fetched_evaluation.task_type == enums.TaskType.CLASSIFICATION

    fetched_evaluation = core.fetch_evaluation_from_job_request(db, job_request_2)
    assert fetched_evaluation.id == evaluation_2
    assert fetched_evaluation.task_type == enums.TaskType.SEGMENTATION


def test_get_evaluation_id_from_job_request(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_1 = core.create_evaluation(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.SEGMENTATION,
    )
    evaluation_2 = core.create_evaluation(db, job_request_2)

    fetched_id = core.get_evaluation_id_from_job_request(db, job_request_1)
    assert fetched_id == evaluation_1

    fetched_id = core.get_evaluation_id_from_job_request(db, job_request_2)
    assert fetched_id == evaluation_2


def test_get_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_1 = core.create_evaluation(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.SEGMENTATION,
    )
    evaluation_2 = core.create_evaluation(db, job_request_2)

    evals = core.get_evaluations(db, evaluation_ids=[evaluation_1, evaluation_2])
    assert len(evals) == 2
    assert enums.TaskType.CLASSIFICATION in [e.task_type for e in evals]
    assert enums.TaskType.SEGMENTATION in [e.task_type for e in evals]


def test_evaluation_status(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_id = core.create_evaluation(db, job_request_1)

    # check that evaluation is created with PENDING status.
    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.PENDING

    # test
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DELETING)

    # set evaluation to running
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.RUNNING)

    # test
    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.RUNNING
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.PENDING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DELETING)

    # set evaluation to failed
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.FAILED)

    # test
    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.FAILED
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.PENDING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # set evaluation to running
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.RUNNING)
    
    # set evaluation to done
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # test
    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.DONE
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.PENDING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.RUNNING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.FAILED)

    # set evaluation to deleting
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DELETING)

    # test
    assert core.get_evaluation_status(db, evaluation_id) == enums.EvaluationStatus.DELETING
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.PENDING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.RUNNING)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.FAILED)


def test_check_for_active_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    evaluation_1 = core.create_evaluation(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.SEGMENTATION,
    )
    evaluation_2 = core.create_evaluation(db, job_request_2)

    # keep evaluation 2 constant, run evaluation 1

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 2

    core.set_evaluation_status(db, evaluation_1, enums.EvaluationStatus.RUNNING)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 2

    core.set_evaluation_status(db, evaluation_1, enums.EvaluationStatus.DONE)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 1

    # create evaluation 3
    job_request_3 = schemas.EvaluationJob(
        dataset=created_dataset,
        model=created_model,
        task_type=enums.TaskType.DETECTION,
    )
    evaluation_3 = core.create_evaluation(db, job_request_3)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 2

    # set both evaluations 2 & 3 to running

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.RUNNING)
    core.set_evaluation_status(db, evaluation_3, enums.EvaluationStatus.RUNNING)

    # test a failed run and then a successful run on evaluation 2

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 2

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.FAILED)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 1

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.RUNNING)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 2

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.DONE)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 1

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.DELETING)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 1

    # finish evaluation 3

    core.set_evaluation_status(db, evaluation_3, enums.EvaluationStatus.DONE)

    assert core.check_for_active_evaluations(db, created_dataset, created_model) == 0