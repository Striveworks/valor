import json
import os
from time import perf_counter

import redis
from sqlalchemy.orm import Session

from velour_api import exceptions, logger, schemas
from velour_api.enums import JobStatus, Stateflow
from velour_api.schemas import BackendStatus, EvaluationJobs

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_SSL = bool(os.getenv("REDIS_SSL"))


# global connection to redis
r: redis.Redis = None


def connect_to_redis():
    global r

    if r is not None:
        return
    try:
        r = redis.Redis(
            REDIS_HOST,
            db=REDIS_DB,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            username=REDIS_USERNAME,
            ssl=REDIS_SSL,
        )
        r.ping()
        logger.info(
            f"succesfully connected to redis instance at {REDIS_HOST}:{REDIS_PORT}"
        )
    except Exception as e:
        logger.debug(
            f"REDIS_HOST: {REDIS_HOST}, REDIS_PORT: {REDIS_PORT}, REDIS_DB: {REDIS_DB}, "
            f"REDIS_PASSWORD: {'null' if REDIS_PASSWORD is None else 'not null'}, "
            f"REDIS_USERNAME: {REDIS_USERNAME}, REDIS_SSL: {REDIS_SSL}"
        )
        raise e


def needs_redis(fn):
    def wrapper(*args, **kwargs):
        if r is None:
            connect_to_redis()
        return fn(*args, **kwargs)

    return wrapper


@needs_redis
def get_evaluation_status(id: int) -> JobStatus:
    json_str = r.get("evaluation_jobs")
    if not isinstance(json_str, str):
        raise RuntimeError("no evaluation jobs exists")
    info = json.loads(json_str)
    jobs = EvaluationJobs(**info)

    if id not in jobs.evaluations:
        raise exceptions.EvaluationJobDoesNotExistError(id)
    return jobs.evaluations[id]


@needs_redis
def set_evaluation_status(id: int, status: JobStatus):
    json_str = r.get("evaluation_jobs")
    if not isinstance(json_str, str):
        jobs = EvaluationJobs(evaluations=dict())
    else:
        info = json.loads(json_str)
        jobs = EvaluationJobs(**info)
    jobs.evaluations[id] = status
    r.set("evaluation_jobs", jobs.model_dump_json())


@needs_redis
def _get_backend_status() -> BackendStatus:
    json_str = r.get("backend_status")
    if not isinstance(json_str, str):
        return BackendStatus()
    info = json.loads(json_str)
    return BackendStatus(**info)


@needs_redis
def _set_backend_status(status: BackendStatus):
    r.set("backend_stateflow", status.model_dump_json())


def _update_backend_status(
    state: Stateflow,
    dataset_name: str,
    model_name: str | None = None,
) -> BackendStatus:

    # get current status
    current_status = _get_backend_status()

    # update status
    if model_name is not None:
        current_status.update_model(
            dataset_name=dataset_name, model_name=model_name, status=state
        )
    else:
        current_status.update_dataset(dataset_name=dataset_name, status=state)

    _set_backend_status(current_status)


def set_dataset_status(dataset_name: str, status: Stateflow):
    _update_backend_status(
        state=status,
        dataset_name=dataset_name,
    )


def set_inference_status(
    dataset_name: str, model_name: str, status: Stateflow
):
    _update_backend_status(
        state=status,
        dataset_name=dataset_name,
        model_name=model_name,
    )


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        if len(args) + len(kwargs) != 2:
            raise RuntimeError

        db = None
        if len(args) > 0:
            db = args[0]
        elif "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        dataset_name = None
        model_name = None
        if len(args) > 1:
            if isinstance(args[1], schemas.Dataset):
                dataset_name = args[1].name
                model_name = None
            elif isinstance(args[1], schemas.Model):
                dataset_name = None
                model_name = args[1].name
            elif isinstance(args[1], schemas.GroundTruth):
                dataset_name = args[1].datum.dataset
                model_name = None
            elif isinstance(args[1], schemas.Prediction):
                dataset_name = args[1].datum.dataset
                model_name = args[1].model
        elif "dataset" in kwargs:
            if isinstance(kwargs["dataset"], schemas.Dataset):
                dataset_name = kwargs["dataset"].name
                model_name = None
        elif "model" in kwargs:
            if isinstance(kwargs["model"], schemas.Model):
                dataset_name = None
                model_name = kwargs["model"].name
        elif "groundtruth" in kwargs:
            if isinstance(kwargs["groundtruth"], schemas.GroundTruth):
                dataset_name = kwargs["groundtruth"].datum.dataset
                model_name = None
        elif "prediction" in kwargs:
            if isinstance(kwargs["prediction"], schemas.Prediction):
                dataset_name = kwargs["prediction"].datum.dataset
                model_name = kwargs["prediction"].model
        else:
            raise RuntimeError

        _update_backend_status(
            Stateflow.CREATE, dataset_name=dataset_name, model_name=model_name
        )
        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        if len(args) + len(kwargs) != 3:
            raise RuntimeError

        db = None
        if len(args) > 0:
            db = args[0]
        elif "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        dataset_name = None
        if len(args) > 1:
            dataset_name = args[1]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]

        model_name = None
        if len(args) > 2:
            model_name = args[2]
        elif "model_name" in kwargs:
            model_name = kwargs["model_name"]

        _update_backend_status(
            state=Stateflow.READY,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        if len(args) + len(kwargs) != 2:
            raise RuntimeError

        db = None
        if len(args) > 0:
            db = args[0]
        elif "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        dataset_name = None
        model_name = None
        if len(args) > 1:
            if isinstance(args[1], schemas.ClfMetricsRequest):
                dataset_name = args[1].settings.dataset
                model_name = args[1].settings.model
            elif isinstance(args[1], schemas.APRequest):
                dataset_name = args[1].settings.dataset
                model_name = args[1].settings.model
        elif "request_info" in kwargs:
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        _update_backend_status(
            Stateflow.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        return fn(*args, **kwargs)

    return wrapper


def computation(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        if len(args) + len(kwargs) != 3:
            raise RuntimeError

        db = None
        if len(args) > 0:
            db = args[0]
        elif "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        dataset_name = None
        model_name = None
        if len(args) > 1:
            if isinstance(args[1], schemas.ClfMetricsRequest):
                dataset_name = args[1].settings.dataset
                model_name = args[1].settings.model
            elif isinstance(args[1], schemas.APRequest):
                dataset_name = args[1].settings.dataset
                model_name = args[1].settings.model
        elif "request_info" in kwargs:
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        _update_backend_status(
            Stateflow.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        result = fn(*args, **kwargs)
        _update_backend_status(
            Stateflow.READY, dataset_name=dataset_name, model_name=model_name
        )

        return result

    return wrapper


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        if len(args) + len(kwargs) != 3:
            raise RuntimeError

        kwargs["_pre_"] = Stateflow.DELETE
        return fn(*args, **kwargs)

    return wrapper
