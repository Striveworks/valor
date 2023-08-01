import json
import os
from time import perf_counter

import redis

from velour_api import logger
from velour_api.enums import JobStatus
from velour_api.exceptions import JobDoesNotExistError
from velour_api.schemas import Job

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
def get_job(uid: str) -> Job:
    json_str = r.get(uid)
    if json_str is None:
        raise JobDoesNotExistError(uid)
    job_info = json.loads(json_str)
    return Job(uid=uid, **job_info)


@needs_redis
def get_all_jobs() -> list[Job]:
    return [get_job(uid) for uid in r.keys()]


@needs_redis
def add_job(job: Job) -> None:
    """Adds job to redis"""
    r.set(job.uid, job.model_dump_json(exclude={"uid"}))


def wrap_method_for_job(
    fn: callable, job_attribute_name_for_output: str | None = None
) -> tuple[Job, callable]:
    """This wraps a method to create and update
    a job stored in redis with its state and output

    Parameters
    ----------
    fn
        the method that computes and stores metrics. This should return an
        id for a MetricParams row in the db

    Returns
    -------
    Job, callable
    """
    job = Job()
    add_job(job)

    def wrapped_method(*args, **kwargs):
        try:
            job.status = JobStatus.PROCESSING
            add_job(job)
            logger.debug(f"starting method {fn} for job {job.uid}")
            start = perf_counter()
            fn_output = fn(*args, **kwargs)
            logger.debug(
                f"method for job {job.uid} finished in {perf_counter() - start} seconds"
            )
            job.status = JobStatus.DONE
            if job_attribute_name_for_output is not None:
                setattr(job, job_attribute_name_for_output, fn_output)
            add_job(job)
        except Exception as e:
            job.status = JobStatus.FAILED
            add_job(job)
            raise e

    return job, wrapped_method


def wrap_metric_computation(fn: callable) -> tuple[Job, callable]:
    """Used for wrapping a metric computation. This will set the resulting
    evaluation_settings_id as an attribute on the job
    """
    return wrap_method_for_job(
        fn=fn, job_attribute_name_for_output="evaluation_settings_id"
    )


@needs_redis
def get_status() -> JobStatus:
    json_str = r.get("stateflow")
    if json_str is None:
        return JobStatus(datasets={})
    info = json.loads(json_str)
    return JobStatus(**info)


@needs_redis
def set_status(status: JobStatus):
    r.set("stateflow", status.model_dump_json())
