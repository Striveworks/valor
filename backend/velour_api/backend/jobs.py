import json
import os

import redis

from velour_api import exceptions, logger
from velour_api.enums import JobStatus
from velour_api.schemas import BackendStateflow, JobStateflow

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


""" Job Status """


@needs_redis
def get_status(id: int) -> JobStatus:
    json_str = r.get("jobs")
    if json_str is None or not isinstance(json_str, bytes):
        raise exceptions.JobDoesNotExistError(id)
    info = json.loads(json_str)
    stateflow = JobStateflow(**info)
    if id not in stateflow.jobs:
        raise exceptions.JobDoesNotExistError(id)
    return stateflow.jobs[id]


@needs_redis
def set_status(id: int, status: JobStatus):
    json_str = r.get("jobs")
    if json_str is None or not isinstance(json_str, bytes):
        stateflow = JobStateflow(jobs=dict())
    else:
        info = json.loads(json_str)
        stateflow = JobStateflow(**info)
    stateflow.set_job(id, status)
    r.set("jobs", stateflow.model_dump_json())


@needs_redis
def remove_status(id: int):
    json_str = r.get("jobs")
    if json_str is None or not isinstance(json_str, bytes):
        raise exceptions.JobDoesNotExistError(id)
    else:
        info = json.loads(json_str)
        stateflow = JobStateflow(**info)
    stateflow.remove_job(id)
    r.set("jobs", stateflow.model_dump_json())


""" Backend Stateflow """


@needs_redis
def get_backend_state() -> BackendStateflow:
    json_str = r.get("backend_stateflow")
    if json_str is None or not isinstance(json_str, bytes):
        return BackendStateflow()
    info = json.loads(json_str)
    return BackendStateflow(**info)


@needs_redis
def set_backend_state(status: BackendStateflow):
    r.set("backend_stateflow", status.model_dump_json())
