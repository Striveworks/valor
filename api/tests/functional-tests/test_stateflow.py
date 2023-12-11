from unittest.mock import MagicMock

import pytest

from velour_api import schemas, enums
from velour_api.crud import jobs
from velour_api.crud.jobs import Job
from velour_api.crud.stateflow import get_job
from velour_api.crud import stateflow
from velour_api.enums import JobStatus
from velour_api.exceptions import (
    DatasetDoesNotExistError,
    DatasetNotFinalizedError,
    ModelNotFinalizedError,
    ModelDoesNotExistError,
    JobStateError,
)

@pytest.fixture(autouse=True)
def db():
    """The setup checks that the redis db is empty and
    the teardown flushes it
    """
    jobs.connect_to_redis()
    if len(jobs.r.keys()) != 0:
        raise RuntimeError("redis database is not-empty")
    yield
    jobs.r.flushdb()


@pytest.fixture
def dataset() -> schemas.Dataset:
    return schemas.Dataset(name="dataset")


@pytest.fixture
def datum(dataset: schemas.Dataset) -> schemas.Datum:
    return schemas.Datum(uid="1234", dataset=dataset.name)


@pytest.fixture
def groundtruth(datum: schemas.Datum) -> schemas.GroundTruth:
    return schemas.GroundTruth(datum=datum, annotations=[])


@pytest.fixture
def model() -> schemas.Model:
    return schemas.Model(name="model")


@pytest.fixture
def prediction(datum, model) -> schemas.Prediction:
    return schemas.Prediction(model=model.name, datum=datum, annotations=[])


@pytest.fixture
def job_request(dataset, model) -> schemas.EvaluationJob:
    return schemas.EvaluationJob(id=1234, model=model.name, dataset=dataset.name, task_type=enums.TaskType.CLASSIFICATION)


def dataset_func(dataset: schemas.Dataset):
    pass


def dataset_name_func(dataset_name: str):
    pass


def model_func(model: schemas.Model):
    pass


def model_name_func(model_name: str):
    pass


def dataset_and_model_name_func(dataset_name: str, model_name: str):
    pass


def groundtruth_func(groundtruth: schemas.GroundTruth):
    pass


def prediction_func(prediction: schemas.Prediction):
    pass


def evaluation_func(job_request: schemas.EvaluationJob):
    pass


def evaluation_func_with_id(job_request: schemas.EvaluationJob, job_id: int):
    pass


def _test_decorator(uuid: str, fn: callable, kwargs: dict):
    assert Job.get_status(uuid) == JobStatus.NONE
    stateflow.create(fn)(**kwargs)
    assert Job.get_status(uuid) == JobStatus.CREATING
    stateflow.finalize(fn)(**kwargs)
    assert Job.get_status(uuid) == JobStatus.DONE
    stateflow.delete(fn)(**kwargs)
    assert Job.get_status(uuid) == JobStatus.NONE


def test_decorator_dataset(db, dataset, groundtruth):
    uuid = Job.generate_uuid(dataset_name=dataset.name)
    _test_decorator(uuid, dataset_name_func, {"dataset_name": dataset.name})
    _test_decorator(uuid, groundtruth_func, {"groundtruth": groundtruth})


def test_decorator_model(db, model):
    uuid = Job.generate_uuid(model_name=model.name)
    _test_decorator(uuid, model_func, {"model": model})
    _test_decorator(uuid, model_name_func, {"model_name": model.name})


def test_decorator_inference(db, dataset, model, prediction, job_request):
    stateflow.finalize(dataset_func)(dataset=dataset)
    stateflow.finalize(model_func)(model=model)

    uuid = Job.generate_uuid(dataset_name=dataset.name, model_name=model.name)
    _test_decorator(uuid, dataset_and_model_name_func, {"dataset_name": dataset.name, "model_name": model.name})
    _test_decorator(uuid, prediction_func, {"prediction": prediction})
    job_request.id = None
    _test_decorator(uuid, evaluation_func, {"job_request": job_request})

    stateflow.delete(dataset_func)(dataset=dataset)
    stateflow.delete(model_func)(model=model)


def test_decorator_evaluation(db, dataset, model, job_request):
    stateflow.finalize(dataset_func)(dataset=dataset)
    stateflow.finalize(model_func)(model=model)
    stateflow.finalize(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    
    job_request.id = 1234
    uuid = Job.generate_uuid(job_request.dataset, job_request.model, job_request.id)
    _test_decorator(uuid, evaluation_func_with_id, {"job_request": job_request, "job_id": job_request.id})

    stateflow.delete(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    stateflow.delete(dataset_func)(dataset=dataset)
    stateflow.delete(model_func)(model=model)


def test_order_of_operations(db, dataset, model, job_request, groundtruth, prediction):

    # dataset -> model -> inference -> evaluation
    stateflow.finalize(dataset_func(dataset=dataset))
    stateflow.finalize(model_func(model=model))
    stateflow.finalize(prediction_func(prediction=prediction))
    stateflow.finalize(evaluation_func_with_id(job_request=job_request, job_id=1234))

    stateflow.delete(evaluation_func_with_id(job_request=job_request, job_id=1234))
    stateflow.delete(prediction_func(prediction=prediction))
    stateflow.delete(dataset_func(dataset=dataset))
    stateflow.delete(model_func(model=model))
    
    # model -> dataset -> inference -> evaluation
    stateflow.finalize(model_func(model=model))
    stateflow.finalize(dataset_func(dataset=dataset))
    stateflow.finalize(prediction_func(prediction=prediction))
    stateflow.finalize(evaluation_func_with_id(job_request=job_request, job_id=1234))

    stateflow.delete(evaluation_func_with_id(job_request=job_request, job_id=1234))
    stateflow.delete(prediction_func(prediction=prediction))
    stateflow.delete(model_func(model=model))
    stateflow.delete(dataset_func(dataset=dataset))

    # dataset -> inference
    stateflow.finalize(dataset_func)(dataset=dataset)
    with pytest.raises(ModelDoesNotExistError):
        stateflow.finalize(prediction_func)(prediction=prediction)
    stateflow.delete(dataset_func)(dataset=dataset)

    # model -> inference
    stateflow.finalize(model_func)(model=model)
    with pytest.raises(DatasetDoesNotExistError):
        stateflow.finalize(prediction_func)(prediction=prediction)
    stateflow.delete(model_func)(model=model)

    # dataset -> evaluation
    stateflow.finalize(dataset_func)(dataset=dataset)
    with pytest.raises(ModelDoesNotExistError):
        stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    stateflow.delete(dataset_func)(dataset=dataset)

    # model -> evaluation
    stateflow.finalize(model_func)(model=model)
    with pytest.raises(DatasetNotFinalizedError):
        stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    stateflow.delete(model_func)(model=model)

    # dataset -> model -> evaluation
    stateflow.finalize(dataset_func)(dataset=dataset)
    stateflow.finalize(model_func)(model=model)
    with pytest.raises(ModelNotFinalizedError):
        stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    stateflow.delete(model_func)(model=model)
    stateflow.delete(dataset_func)(dataset=dataset)


def test_deletion(db, dataset, model, prediction, job_request):

    def _create_entries():
        # dataset -> model -> inference -> evaluation
        stateflow.finalize(dataset_func)(dataset=dataset)
        stateflow.finalize(model_func)(model=model)
        stateflow.finalize(prediction_func)(prediction=prediction)
        stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=1234)

    # evaluation -> inference -> model -> dataset
    _create_entries()
    stateflow.delete(evaluation_func_with_id)(job_request=job_request, job_id=1234)
    stateflow.delete(prediction_func)(prediction=prediction)
    stateflow.delete(model_func)(model=model)
    stateflow.delete(dataset_func)(dataset=dataset)
    
    # evaluation -> inference -> dataset -> model
    _create_entries()
    stateflow.delete(evaluation_func_with_id)(job_request=job_request, job_id=1234)
    stateflow.delete(prediction_func)(prediction=prediction)
    stateflow.delete(dataset_func)(dataset=dataset)
    stateflow.delete(model_func)(model=model)


def test_creater(db, dataset, model, job_request):
    
    # create dataset
    stateflow.create(dataset_func)(dataset=dataset)
    with pytest.raises(ModelDoesNotExistError):
        stateflow.create(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    
    # create model
    stateflow.create(model_func)(model=model)
    with pytest.raises(ModelDoesNotExistError):
        stateflow.create(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    
    # finalize dataset
    stateflow.finalize(dataset_func)(dataset=dataset)
    with pytest.raises(ModelDoesNotExistError):
        stateflow.create(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)

    # finalize model
    stateflow.finalize(model_func)(model=model)

    # create inference predictions
    stateflow.create(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    with pytest.raises(ModelNotFinalizedError):
        stateflow.create(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)

    # finalize model predictions
    stateflow.finalize(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    stateflow.create(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)

    # finish evaluations
    stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)

    # delete all
    stateflow.delete(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    stateflow.delete(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    stateflow.delete(dataset_func)(dataset=dataset)
    stateflow.delete(model_func)(model=model)


def test_get_job(db, dataset, model, job_request):
    
    # create dataset
    stateflow.create(dataset_func)(dataset=dataset)
    get_job(dataset_name=dataset.name)
    with pytest.raises(ModelDoesNotExistError):
        get_job(dataset_name=dataset.name, model_name=model.name)
    with pytest.raises(DatasetNotFinalizedError):
        get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    # create model
    stateflow.create(model_func)(model=model)
    get_job(dataset_name=dataset.name)
    get_job(model_name=model.name)
    with pytest.raises(ModelDoesNotExistError):
        get_job(dataset_name=dataset.name, model_name=model.name)
    with pytest.raises(DatasetNotFinalizedError):
        get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    # create model 
    stateflow.finalize(model_func)(model=model)
    get_job(dataset_name=dataset.name)
    get_job(model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name)
    with pytest.raises(DatasetNotFinalizedError):
        get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)
    
    # finalize dataset
    stateflow.finalize(dataset_func)(dataset=dataset)
    get_job(dataset_name=dataset.name)
    get_job(dataset_name=dataset.name, model_name=model.name)
    with pytest.raises(JobStateError):
        get_job(model_name=model.name)
    with pytest.raises(ModelNotFinalizedError):
        get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    # create model predictions
    stateflow.create(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name)
    get_job(dataset_name=dataset.name)
    with pytest.raises(JobStateError):
        get_job(model_name=model.name)
    with pytest.raises(ModelNotFinalizedError):
        get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    # finalize model predictions
    stateflow.finalize(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    get_job(dataset_name=dataset.name)  
    get_job(model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    stateflow.create(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)
    with pytest.raises(JobStateError):
        get_job(dataset_name=dataset.name)
    with pytest.raises(JobStateError):
        get_job(model_name=model.name)
    with pytest.raises(JobStateError):
        get_job(dataset_name=dataset.name, model_name=model.name)

    # finish evaluations
    stateflow.finalize(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    get_job(dataset_name=dataset.name)  
    get_job(model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name)
    get_job(dataset_name=dataset.name, model_name=model.name, evaluation_id=job_request.id)

    # delete all
    stateflow.delete(evaluation_func_with_id)(job_request=job_request, job_id=job_request.id)
    stateflow.delete(dataset_and_model_name_func)(dataset_name=dataset.name, model_name=model.name)
    stateflow.delete(dataset_func)(dataset=dataset)
    stateflow.delete(model_func)(model=model)