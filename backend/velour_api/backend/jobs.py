import json
import os
from time import perf_counter

import redis

from velour_api import enums, exceptions, logger
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
def get_evaluation_status(id: int) -> enums.JobStatus:
    json_str = r.get("evaluation_jobs")
    if not isinstance(json_str, str):
        raise RuntimeError("no evaluation jobs exists")
    info = json.loads(json_str)
    jobs = EvaluationJobs(**info)

    if id not in jobs.evaluations:
        raise exceptions.EvaluationJobDoesNotExistError(id)
    return jobs.evaluations[id]


@needs_redis
def set_evaluation_status(id: int, status: enums.JobStatus):
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


def _parse_args(*args, **kwargs) -> tuple[str, str | None]:

    # unpack dataset_name
    if "dataset_name" in kwargs:
        dataset_name = kwargs["dataset_name"]
    elif len(args) > 0:
        if not isinstance(args[0], str):
            raise ValueError("dataset_name must be of type `str`")
        dataset_name = args[0]
    else:
        raise ValueError("dataset_name not provided")

    # unpack model_name
    if "model_name" in kwargs:
        model_name = kwargs["model_name"]
    elif len(args) > 1:
        if not isinstance(args[1], str):
            raise ValueError("model_name must be of type `str`")
        model_name = args[1]
    else:
        model_name = None

    return (dataset_name, model_name)


def _validate_backend_status(
    state: enums.Stateflow,
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

    return current_status


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = _parse_args(*args, **kwargs)

        # validate and update status
        _set_backend_status(
            _validate_backend_status(
                state=enums.Stateflow.CREATE,
                dataset_name=dataset_name,
                model_name=model_name,
            )
        )

        # execute wrapped method
        result = fn(*args, **kwargs)

        return result

    return wrapper


def read(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = _parse_args(*args, **kwargs)

        # validate status for read
        status = _get_backend_status()
        if not status.readable(
            dataset_name=dataset_name, model_name=model_name
        ):
            raise exceptions.StateflowError(
                "unable to read as a delete operation is in progress"
            )

        # execute wrapped method
        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = _parse_args(*args, **kwargs)

        # validate and update status
        _set_backend_status(
            _validate_backend_status(
                state=enums.Stateflow.READY,
                dataset_name=dataset_name,
                model_name=model_name,
            )
        )

        # execute wrapped method
        result = fn(*args, **kwargs)

        return result

    return wrapper


def evaluate(persist: bool = False) -> callable:
    """ """

    def decorator(fn: callable):
        def wrapper(*args, **kwargs):
            # unpack arguments
            dataset_name, model_name = _parse_args(*args, **kwargs)

            # validate status
            _set_backend_status(
                _validate_backend_status(
                    state=enums.Stateflow.EVALUATE,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )
            )

            # execute wrapped method
            result = fn(*args, **kwargs)

            # conditional: persist the state after method is executed
            if not persist:
                _set_backend_status(
                    _validate_backend_status(
                        state=enums.Stateflow.READY,
                        dataset_name=dataset_name,
                        model_name=model_name,
                    )
                )

            return result

        return wrapper

    return decorator


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = _parse_args(*args, **kwargs)

        # get status
        status = _validate_backend_status(
            state=enums.Stateflow.DELETE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        # execute wrapped method
        ret = fn(*args, **kwargs)

        # clear status
        if model_name is not None:
            status.remove_model(
                dataset_name=dataset_name, model_name=model_name
            )
        else:
            status.remove_dataset(dataset_name=dataset_name)

        # update backend status
        _set_backend_status(status)

        return ret

    return wrapper


def debug(fn: callable) -> callable:
    """This wraps a method with a debug timer

    Parameters
    ----------
    fn
        any method

    Returns
    -------
    callable
    """

    def wrapped_method(*args, **kwargs):
        try:
            logger.debug(f"starting method {fn}")
            start = perf_counter()

            ret = fn(*args, **kwargs)

            logger.debug(
                f"method finished in {perf_counter() - start} seconds"
            )
        except Exception as e:
            raise e

        return ret

    return wrapped_method
