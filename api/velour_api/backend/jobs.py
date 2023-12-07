import json
import os
import time
from functools import wraps
from pydantic import BaseModel

import redis

from velour_api import logger, schemas
from velour_api.enums import JobStatus

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_SSL = bool(os.getenv("REDIS_SSL"))

# global connection to redis
r: redis.Redis = None


def retry_connection(timeout: int):
    """
    Decorator to help retry a connection.

    Parameters
    ----------
    timeout : int
        The number of seconds to wait before throwing an exception.

    RuntimeError
    ------
    HTTPException (404)
        If the the job doesn't succeed before the timeout parameter.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if time.time() - start_time >= timeout:
                        logger.debug(
                            f"REDIS_HOST: {REDIS_HOST}, REDIS_PORT: {REDIS_PORT}, REDIS_DB: {REDIS_DB}, "
                            f"REDIS_PASSWORD: {'null' if REDIS_PASSWORD is None else 'not null'}, "
                            f"REDIS_USERNAME: {REDIS_USERNAME}, REDIS_SSL: {REDIS_SSL}"
                        )
                        raise RuntimeError(
                            f"Method {func.__name__} failed to connect to database within {timeout} seconds, with error: {str(e)}"
                        )
                time.sleep(2)

        return wrapper

    return decorator


@retry_connection(30)
def connect_to_redis():
    """
    Connect to the redis service.
    """

    global r
    if r is not None:
        return

    connection = redis.Redis(
        REDIS_HOST,
        db=REDIS_DB,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        ssl=REDIS_SSL,
    )
    connection.ping()  # this should fail if redis does not exist
    logger.info(
        f"succesfully connected to redis instance at {REDIS_HOST}:{REDIS_PORT}"
    )

    # Only set if connected
    r = connection


def needs_redis(fn):
    def wrapper(*args, **kwargs):
        if r is None:
            connect_to_redis()
        return fn(*args, **kwargs)

    return wrapper


class Job(BaseModel):
    uuid: str
    status: JobStatus = JobStatus.PENDING
    msg: str = ""
    children: set[str] = {}

    @classmethod
    @needs_redis
    def get(
        cls, 
        uuid: str = None,
    ):
        json_str = r.get(uuid)
        if json_str is None or not isinstance(json_str, bytes):
            job = cls(uuid=uuid)
            r.set(uuid, job.model_dump_json(exclude={'uuid'}))
            return job
        job = json.loads(json_str)
        job["uuid"] = uuid
        return cls(**job)
    
    @classmethod
    @needs_redis
    def retrieve(
        cls, 
        dataset_name: str = None,
        model_name: str = None,
        evaluation_id: int = None,
    ):
        uuid = generate_uuid(dataset_name, model_name, evaluation_id)
        return cls.get(uuid)

    @needs_redis
    def set(self):
        r.set(self.uuid, self.model_dump_json(exclude={'uuid'}))
    
    def set_status(self, status: JobStatus, msg: str = ""):
        if status not in self.status.next():
            raise ValueError(f"{status} not in {self.status.next()}")
        self.status = status
        self.msg = msg
        self.set()

    def register_child(self, uuid: int):
        self.children.add(uuid)
        self.set()


def generate_uuid(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> int:
    if not (dataset_name or model_name or evaluation_id):
        raise ValueError
    return (f"{dataset_name}+{model_name}+{evaluation_id}")
