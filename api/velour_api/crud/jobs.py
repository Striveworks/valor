import json
import os
import time
from functools import wraps
from pydantic import BaseModel, Field

import redis

from velour_api import logger, schemas
from velour_api.enums import JobStatus
from velour_api.exceptions import JobStateError

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
    children: set[str] = Field(default_factory=set)

    @classmethod
    @needs_redis
    def get(
        cls, 
        uuid: str,
    ):
        """
        Create or get Job with matching UUID.
        """
        json_str = r.get(uuid)
        if json_str is None or not isinstance(json_str, bytes):
            job = cls(uuid=uuid)
            r.set(uuid, job.model_dump_json(exclude={'uuid'}))
            return job
        job = json.loads(json_str)
        job["uuid"] = uuid
        return cls(**job)
    
    @needs_redis
    def sync(self):
        """
        Set redis to match Job object.
        """
        r.set(self.uuid, self.model_dump_json(exclude={'uuid'}))
    
    def set_status(self, status: JobStatus, msg: str = ""):
        """
        Set job status.
        """
        if status not in self.status.next():
            raise JobStateError(self.uuid, f"{status} not in {self.status} next set: {self.status.next()}")
        self.status = status
        self.msg = msg
        self.sync()

    @needs_redis
    def get_status(self) -> JobStatus:
        """Fetch job status from redis."""
        json_str = r.get(self.uuid)
        if json_str is None or not isinstance(json_str, bytes):
            return JobStatus.NONE
        job = json.loads(json_str)
        return JobStatus(job["status"])

    def register_child(self, uuid: int):
        """
        Register a child Job.
        """
        self.children.add(uuid)
        self.sync()

    @needs_redis
    def delete(self):
        """Delete job from redis."""
        for child_uuid in self.children:
            if self.get_status(uuid=child_uuid) not in [JobStatus.NONE, JobStatus.DONE]:
                raise JobStateError(self.uuid, f"Job blocked by child task with uuid `{child_uuid}` and status `{Job.get_status(uuid=child_uuid).value}`")
        for child_uuid in self.children:
            self.get(child_uuid).delete()
        r.delete(self.uuid)
        del self


def generate_uuid(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> str:
    """
    Generate a UUID from a combination of the input args.

    Input must match one of the following sets: {dataset_name}, {model_name}, {dataset_name, model_name} or {dataset_name, model_name, evaluation_id}.

    Parameters
    ----------
    dataset_name : Optional[str]
        Dataset name.
    datum_uid: Optional[str]
        Datum uid.
    model_name : Optional[str]
        Model name.
    evaluation_id : Optional[int]
        Evaluation id.

    Returns
    ----------
    int
    """
    return (f"{dataset_name}+{model_name}+{evaluation_id}")


@needs_redis
def get_status_from_uuid(uuid: str):
    """Fetch job status from redis."""
    json_str = r.get(uuid)
    if json_str is None or not isinstance(json_str, bytes):
        return JobStatus.NONE
    job = json.loads(json_str)
    return JobStatus(job["status"])

    
@needs_redis
def get_status_from_names(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
):
    if evaluation_id and not (dataset_name or model_name):
        uuids = r.keys(pattern=f"*+*+{evaluation_id}")
        if not uuids:
            return JobStatus.NONE
        uuid = uuids[0].decode("utf-8")
    else:
        uuid = generate_uuid(dataset_name, model_name, evaluation_id)
    return get_status_from_uuid(uuid)



j = Job(uuid="ds+md+123", status=JobStatus.CREATING)
j.set()
j.set_status(JobStatus.DONE)
print(Job.get_uuid(evaluation_id=123))

