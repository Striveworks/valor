import json
import os
import time
from functools import wraps

import redis
from pydantic import BaseModel, Field

from velour_api import logger
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

    Raises
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
    """
    Decorator to ensure that Redis is connected before wrapped function is executed.
    """

    def wrapper(*args, **kwargs):
        if r is None:
            connect_to_redis()
        return fn(*args, **kwargs)

    return wrapper


def generate_uuid(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> str:
    """
    Generate a UUID based on a combination of input arguments.

    The UUID is created from the provided parameters and must conform
    to one of the following sets to properly generate a UUID.

    - Dataset Job: {dataset_name}
    - Model Job: {model_name}
    - Model Prediction Job: {dataset_name, model_name}
    - Evaluation Job: {dataset_name, model_name, evaluation_id}

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset.
    model_name : str, optional
        Name of the model.
    evaluation_id : int, optional
        Unique identifier for the evaluation.

    Returns
    -------
    str
        A unique job identifier generated from the input parameters.
    """
    return f"{dataset_name}+{model_name}+{evaluation_id}"


@needs_redis
def get_status_from_uuid(uuid: str) -> JobStatus:
    """
    Fetch a job using its UUID and return its status.

    If no matching Job exists, JobStatus.NONE is returned.

    Parameters
    ----------
    uuid : str
        Job UUID.

    Returns
    -------
    velour.enums.JobStatus
        The status of the fetched job, or JobStatus.NONE if no matching job is found.
    """
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
) -> JobStatus:
    """
    Fetch a job and return its status based on the provided input.

    The input must conform to one of the following sets to properly fetch a job:

    - Dataset Job: {dataset_name}
    - Model Job: {model_name}
    - Model Prediction Job: {dataset_name, model_name}
    - Evaluation Job: {evaluation_id} or {dataset_name, model_name, evaluation_id}

    If no matching Job exists, JobStatus.NONE is returned.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset.
    model_name : str, optional
        Name of the model.
    evaluation_id : int, optional
        Unique identifier for the evaluation.

    Returns
    -------
    velour.enums.JobStatus
        The status of the fetched job, or JobStatus.NONE if no matching job is found.
    """
    if evaluation_id and not (dataset_name or model_name):
        uuids = r.keys(pattern=f"*+*+{evaluation_id}")
        if not uuids:
            return JobStatus.NONE
        uuid = uuids[0].decode("utf-8")
    else:
        uuid = generate_uuid(dataset_name, model_name, evaluation_id)
    return get_status_from_uuid(uuid)


class Job(BaseModel):
    """
    Job is a database abstraction layer.

    This class serves as an object wrapper for interacting with job-related data,
    utilizing Redis as the default database. However, it can be easily adapted
    to support other databases.

    Attributes
    ----------
    uuid : str
        A unique identifier for the job.
    status : velour_api.enums.JobStatus, default=JobStatus.PENDING
        The status of the job.
    msg : str, default=""
        Additional information or messages related to the job.
    children : set[str], default=Field(default_factory=set)
        A set containing identifiers of child jobs associated with this job.
    """

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
        Create or fetch a Job with the specified UUID from Redis.

        Parameters
        ----------
        uuid : str
            The UUID of the Job to be retrieved or created.

        Returns
        -------
        Job
            A Job object with the specified UUID.
        """
        json_str = r.get(uuid)
        if json_str is None or not isinstance(json_str, bytes):
            job = cls(uuid=uuid)
            r.set(uuid, job.model_dump_json(exclude={"uuid"}))
            return job
        job = json.loads(json_str)
        job["uuid"] = uuid
        return cls(**job)

    @needs_redis
    def sync(self):
        """
        Synchronize the Job object with Redis.
        """
        r.set(self.uuid, self.model_dump_json(exclude={"uuid"}))

    def set_status(self, status: JobStatus, msg: str = ""):
        """
        Set the job status with an optional message.

        Parameters
        ----------
        status : velour_api.enums.JobStatus
            The new job status.
        msg : str, default=""
            An optional message to include with the new status.

        Raises
        ------
        JobStateError
            If the specified status is not in the allowed next states based on
            the current job status.
        """
        if status not in self.status.next():
            raise JobStateError(
                self.uuid,
                f"{status} not in {self.status} next set: {self.status.next()}",
            )
        self.status = status
        self.msg = msg
        self.sync()

    @needs_redis
    def get_status(self) -> JobStatus:
        """
        Retrieve the status of the job from Redis.

        Returns
        -------
        velour.enums.JobStatus
            The current status of the job.
        """
        return get_status_from_uuid(self.uuid)

    def register_child(self, uuid: str):
        """
        Register a child Job by adding its UUID to the set of children.

        Parameters
        ----------
        uuid : str
            The UUID of the child Job to be registered.
        """
        self.children.add(uuid)
        self.sync()

    @needs_redis
    def delete(self):
        """
        Delete the job and any associated child jobs from Redis.
        """
        for child_uuid in self.children:
            self.get(child_uuid).delete()
        r.delete(self.uuid)
