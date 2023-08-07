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
    if json_str is None:
        raise RuntimeError("no evaluation jobs exists")
    info = json.loads(json_str)
    jobs = EvaluationJobs(**info)

    if id not in jobs.evaluations:
        raise exceptions.EvaluationJobDoesNotExistError(id)
    return jobs.evaluations[id]


@needs_redis
def set_evaluation_status(id: int, status: JobStatus):
    json_str = r.get("evaluation_jobs")
    if json_str is None:
        jobs = EvaluationJobs(evaluations=dict())
    else:
        info = json.loads(json_str)
        jobs = EvaluationJobs(**info)
    jobs.evaluations[id] = status
    r.set("evaluation_jobs", jobs.model_dump_json())


@needs_redis
def _get_backend_status() -> BackendStatus:
    json_str = r.get("backend_stateflow")
    if json_str is None:
        return BackendStatus()
    info = json.loads(json_str)
    return BackendStatus(**info)


@needs_redis
def _set_backend_status(status: BackendStatus):
    r.set("backend_stateflow", status.model_dump_json())


def _update_backend_status(
    state: Stateflow,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> BackendStatus:

    # get current status
    current_status = _get_backend_status()

    # update status
    if model_name:
        current_status.set_model_status(
            status=state,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    elif dataset_name:
        current_status.set_dataset_status(
            dataset_name=dataset_name, status=state
        )

    _set_backend_status(current_status)


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack db
        db = None
        if "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None

        if "groundtruth" in kwargs:
            if isinstance(kwargs["groundtruth"], schemas.GroundTruth):
                dataset_name = kwargs["groundtruth"].datum.dataset
                model_name = None
        elif "prediction" in kwargs:
            if isinstance(kwargs["prediction"], schemas.Prediction):
                dataset_name = kwargs["prediction"].datum.dataset
                model_name = kwargs["prediction"].model

        if dataset_name is not None:
            _update_backend_status(
                Stateflow.CREATE,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 3:
            raise RuntimeError

        # unpack db
        db = None
        if "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        # unpack dataset
        dataset_name = None
        if "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]

        # unpack model
        model_name = None
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]

        # enter ready state
        _update_backend_status(
            state=Stateflow.READY,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack db
        db = None
        if "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None
        if "request_info" in kwargs:
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        # put model / dataset in evaluation state
        _update_backend_status(
            Stateflow.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        return fn(*args, **kwargs)

    return wrapper


def computation(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack db
        db = None
        if "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None
        if "request_info" in kwargs:
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        # start eval computation
        _update_backend_status(
            Stateflow.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        result = fn(*args, **kwargs)

        # end eval computation
        _update_backend_status(
            Stateflow.READY, dataset_name=dataset_name, model_name=model_name
        )

        return result

    return wrapper


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 3:
            raise RuntimeError

        # unpack db
        db = None
        if "db" in kwargs:
            db = kwargs["db"]
        if not isinstance(db, Session):
            raise RuntimeError

        # unpack dataset
        dataset_name = None
        if "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]

        # unpack model
        model_name = None
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]

        # enter deletion state
        _update_backend_status(
            state=Stateflow.DELETE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        result = fn(*args, **kwargs)

        # remove
        status = _get_backend_status()
        if model_name is not None:
            status.remove_model(model_name)
        else:
            status.remove_dataset(dataset_name)
        _set_backend_status(status)

        return result

    return wrapper


def debug_timer(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        logger.debug(f"starting method {fn}")
        start = perf_counter()
        result = fn(*args, **kwargs)
        logger.debug(
            f"method {fn} finished in {perf_counter() - start} seconds"
        )
        return result

    return wrapper
