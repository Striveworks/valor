import json
import os
from time import perf_counter

import redis

from velour_api import logger
from velour_api.enums import JobStatus
from velour_api.exceptions import JobDoesNotExistError
from velour_api.schemas import EvalJob

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")

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
        )
        r.ping()
        logger.info(
            f"succesfully connected to redis instance at {REDIS_HOST}:{REDIS_PORT}"
        )
    except Exception as e:
        logger.debug(
            f"error connecting to redis instance at {REDIS_HOST}:{REDIS_PORT} "
            f"with username {REDIS_USERNAME} and password {'null' if REDIS_PASSWORD is None else 'not null'}"
        )
        raise e


def needs_redis(fn):
    def wrapper(*args, **kwargs):
        if r is None:
            connect_to_redis()
        return fn(*args, **kwargs)

    return wrapper


@needs_redis
def get_job(uid: str) -> EvalJob:
    json_str = r.get(uid)
    if json_str is None:
        raise JobDoesNotExistError(uid)
    job_info = json.loads(json_str)
    return EvalJob(uid=uid, **job_info)


@needs_redis
def get_all_jobs() -> list[EvalJob]:
    return [get_job(uid) for uid in r.keys()]


@needs_redis
def add_job(job: EvalJob) -> None:
    """Adds job to redis"""
    r.set(job.uid, job.json(exclude={"uid"}))


def wrap_metric_computation(fn: callable) -> tuple[EvalJob, callable]:
    """This wraps a metric computation method to create and update
    a job stored in redis with its state and output

    Parameters
    ----------
    fn
        the method that computes and stores metrics. This should return an
        id for a MetricParams row in the db

    Returns
    -------
    EvalJob, callable
        returns the created job and the wrapped method
    """
    job = EvalJob()
    add_job(job)

    def wrapped_method(*args, **kwargs):
        try:
            job.status = JobStatus.PROCESSING
            add_job(job)
            logger.debug(f"starting computing metrics using {fn}")
            start = perf_counter()
            metric_params_id = fn(*args, **kwargs)
            logger.debug(
                f"finished computing metrics in {perf_counter() - start} seconds"
            )
            job.status = JobStatus.DONE
            job.metric_params_id = metric_params_id
            add_job(job)
        except Exception as e:
            job.status = JobStatus.FAILED
            add_job(job)
            raise e

    return job, wrapped_method
