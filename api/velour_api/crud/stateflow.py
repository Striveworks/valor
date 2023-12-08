from functools import wraps

from velour_api import logger, schemas
from velour_api.crud.jobs import Job
from velour_api.enums import JobStatus
from velour_api.exceptions import (
    JobStateError, 
    DatasetDoesNotExistError,
    DatasetNotFinalizedError,
    ModelDoesNotExistError,
    ModelNotFinalizedError, 
)


def get_job(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
):
    job = None

    dataset_uuid = Job.generate_uuid(dataset_name=dataset_name)
    model_uuid = Job.generate_uuid(model_name=model_name)
    inference_uuid = Job.generate_uuid(dataset_name=dataset_name, model_name=model_name)
    evaluation_uuid = Job.generate_uuid(dataset_name=dataset_name, model_name=model_name, evaluation_id=evaluation_id)
    
    # initialize and check parent states
    if evaluation_id and dataset_name and model_name:
        if Job.get_status(dataset_uuid) != JobStatus.DONE:
            raise DatasetNotFinalizedError(name=dataset_name)
        elif Job.get_status(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        elif Job.get_status(inference_uuid) != JobStatus.DONE:
            raise ModelNotFinalizedError(dataset_name=dataset_name, model_name=model_name)
        job = Job.get(evaluation_uuid)
        Job.get(inference_uuid).register_child(job.uuid)
        Job.get(dataset_uuid).register_child(job.uuid)
    elif dataset_name and model_name:
        if Job.get_status(dataset_uuid) not in [JobStatus.PROCESSING, JobStatus.DONE]:
            raise DatasetDoesNotExistError(name=dataset_name)
        elif Job.get_status(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        job = Job.get(inference_uuid)
        Job.get(model_uuid).register_child(job.uuid)    
    elif dataset_name:
        job = Job.get(dataset_uuid)
    elif model_name:
        job = Job.get(model_uuid)
    else:
        raise ValueError(f"Received invalid input.")
    
    def _recursive_child_search(job: Job):
        # Check status of child jobs
        for uuid in job.children:
            status = Job.get_status(uuid=uuid)
            if status not in [JobStatus.NONE, JobStatus.DONE]:
                raise JobStateError(job.uuid, f"Job blocked by child task with uuid `{uuid}` and status `{Job.get_status(uuid=uuid).value}`")
            elif status == JobStatus.DONE:
                _recursive_child_search(Job.get(uuid))
    _recursive_child_search(job)
    
    return job


def get_status(        
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> JobStatus:
    uuid = Job.generate_uuid(dataset_name, model_name, evaluation_id)
    return Job.get_status(uuid)


def custom(
    on_start: callable = lambda job, msg="" : job.set_status(JobStatus.PROCESSING, msg),
    on_success: callable = lambda job, msg="" : job.set_status(JobStatus.DONE, msg),
    on_failure: callable = lambda job, msg="" : job.set_status(JobStatus.FAILED, msg),
):
    def decorator(fn: callable) -> callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                raise ValueError("Positional args not supported.")
            dataset_name = None
            model_name = None
            evaluation_id = None
            if "dataset" in kwargs:
                dataset_name = kwargs["dataset"].name
            elif "model" in kwargs:
                model_name = kwargs["model"].name
            elif "groundtruth" in kwargs:
                dataset_name = kwargs["groundtruth"].datum.dataset
            elif "prediction" in kwargs:
                dataset_name = kwargs["prediction"].datum.dataset
                model_name = kwargs["prediction"].model
            elif "job_request" in kwargs:
                dataset_name = kwargs["job_request"].dataset
                model_name = kwargs["job_request"].model                
                if "job_id" in kwargs:
                    evaluation_id = kwargs["job_request"].id
            elif "dataset_name" in kwargs and "model_name" in kwargs:
                dataset_name = kwargs["dataset_name"]
                model_name = kwargs["model_name"]
            elif "dataset_name" in kwargs:
                dataset_name = kwargs["dataset_name"]
            elif "model_name" in kwargs:
                model_name = kwargs["model_name"]
            else:
                raise ValueError("did not receive right values")

            job = get_job(dataset_name, model_name, evaluation_id)
            on_start(job)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                on_failure(job, str(e))
                raise e
            on_success(job)
            return result
        return wrapper
    return decorator


# stateflow decorators
initialize = custom(
    on_start=lambda job, msg="" : job.set_status(JobStatus.PROCESSING, msg),
    on_success=lambda job, msg="" : job.set_status(JobStatus.PROCESSING, msg),
)
run = custom()
delete = custom(
    on_start=lambda job, msg="" : job.set_status(JobStatus.DELETING),
    on_success=lambda job, msg="" : job.delete(),
)

