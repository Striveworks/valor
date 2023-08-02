import json
import os
from time import perf_counter

import redis

from velour_api import enums, exceptions, logger
from velour_api.schemas import BackendStatus, EvaluationStatus

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
def get_backend_status() -> BackendStatus:
    json_str = r.get("backend_status")
    if json_str is None:
        return BackendStatus()
    info = json.loads(json_str)
    return BackendStatus(**info)


@needs_redis
def set_backend_status(status: BackendStatus):
    r.set("backend_stateflow", status.model_dump_json())


@needs_redis
def get_evaluation_status() -> EvaluationStatus:
    json_str = r.get("evaluation_status")
    if json_str is None:
        return EvaluationStatus()
    info = json.loads(json_str)
    return EvaluationStatus(**info)


@needs_redis
def set_evaluation_status(status: EvaluationStatus):
    r.set("backend_stateflow", status.model_dump_json())


def get_names(*args, **kwargs) -> tuple[str, str | None]:

    # unpack dataset_name
    if "dataset_name" in kwargs:
        dataset_name = kwargs["dataset_name"]
    elif len(args) > 0:
        dataset_name = args[0]
    else:
        raise ValueError("dataset_name not provided")

    # unpack model_name
    if "model_name" in kwargs:
        model_name = kwargs["model_name"]
    elif len(args) > 1:
        model_name = args[1]
    else:
        model_name = None

    return (dataset_name, model_name)


def update_backend_status(
    state: enums.Stateflow,
    dataset_name: str,
    model_name: str | None = None,
) -> BackendStatus:

    # get status
    status = get_backend_status()

    # update status
    if state == enums.Stateflow.CREATE:
        if model_name is not None:
            status.add_model(dataset_name=dataset_name, model_name=model_name)
        else:
            status.add_dataset(dataset_name=dataset_name)
    else:
        if model_name is not None:
            status.update_model(
                dataset_name=dataset_name, model_name=model_name, status=state
            )
        else:
            status.update_dataset(dataset_name=dataset_name, status=state)

    return status


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = get_names(*args, **kwargs)

        # get status
        status = update_backend_status(
            state=enums.Stateflow.CREATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        # execute wrapped method
        return fn(*args, **kwargs)

        # update backend status
        set_backend_status(status)

    return wrapper


def read(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = get_names(*args, **kwargs)

        # validate status for read
        status = get_backend_status()
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
        dataset_name, model_name = get_names(*args, **kwargs)

        # get/update state
        status = update_backend_status(
            state=enums.Stateflow.READY,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        # execute wrapped method
        ret = fn(*args, **kwargs)

        # update backend status
        set_backend_status(status)

        return ret

    return wrapper


def evaluate(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = get_names(*args, **kwargs)

        # get status
        status = update_backend_status(
            state=enums.Stateflow.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        # execute wrapped method
        ret = fn(*args, **kwargs)

        # update backend status
        set_backend_status(status)

        return ret

    return wrapper


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # unpack arguments
        dataset_name, model_name = get_names(*args, **kwargs)

        # get status
        status = update_backend_status(
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
        set_backend_status(status)

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
