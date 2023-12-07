from functools import wraps

from velour_api import logger, schemas
from velour_api.backend.jobs import Job, generate_uuid
from velour_api.enums import JobStatus


def get_job(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
):
    job = None
    
    # initialize and check parent states
    if dataset_name and model_name and evaluation_id:
        if (
            Job.retrieve(dataset_name=dataset_name).status != JobStatus.DONE
            or Job.retrieve(model_name=model_name).status != JobStatus.DONE
            or Job.retrieve(dataset_name=dataset_name, model_name=model_name).status != JobStatus.DONE
        ):
            raise RuntimeError
        job = Job.retrieve(dataset_name=dataset_name, model_name=model_name, evaluation_id=evaluation_id)
        Job.retrieve(dataset_name=dataset_name, model_name=model_name).register_child(job.uuid)
    elif dataset_name and model_name:
        if (
            Job.retrieve(dataset_name=dataset_name).status != JobStatus.DONE
            or Job.retrieve(model_name=model_name).status != JobStatus.DONE
        ):
            raise RuntimeError
        job = Job.retrieve(dataset_name=dataset_name, model_name=model_name)
        Job.retrieve(dataset_name=dataset_name).register_child(job.uuid)
        Job.retrieve(model_name=model_name).register_child(job.uuid)
    elif dataset_name:
        job = Job.retrieve(dataset_name=dataset_name)
    elif model_name:
        job = Job.retrieve(model_name=model_name)
    else:
        raise RuntimeError
    
    # Check status of child jobs
    for uuid in job.children:
        if Job.get(uuid=uuid).status != JobStatus.DONE:
            raise RuntimeError
        
    return job


def custom(
    onStart: JobStatus,
    onCompletion: JobStatus,
    onFailure: JobStatus = JobStatus.FAILED,
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
            job.set_status(onStart)
            try:
                fn(*args, **kwargs)
            except Exception as e:
                job.set_status(onFailure, str(e))
                raise e
            job.set_status(onCompletion)
        return wrapper
    return decorator


default = custom(
    onStart=JobStatus.PROCESSING,
    onCompletion=JobStatus.DONE,
)

create = custom(
    onStart=JobStatus.PENDING,
    onCompletion=JobStatus.PENDING,
)

finalize = custom(
    onStart=JobStatus.PROCESSING,
    onCompletion=JobStatus.DONE,
)

import time
from velour import enums

@create
def some_func1(model: schemas.Model):
    time.sleep(1)

@finalize
def some_func2(model: schemas.Model):
    time.sleep(1)

@default
def other_func(dataset: schemas.Dataset):
    time.sleep(1)

@default
def another_func(job_request: schemas.EvaluationJob):
    time.sleep(1)


if __name__ == "__main__":

    model = schemas.Model(name="model")
    dataset = schemas.Dataset(name="dataset")
    job_request = schemas.EvaluationJob(model=model.name, dataset=dataset.name, task_type=enums.TaskType.CLASSIFICATION)
    
    some_func1(model=model)
    other_func(dataset=dataset)
    other_func(dataset=dataset)
    # another_func(job_request=job_request)
    some_func2(model=model)
    another_func(job_request=job_request)
    other_func(dataset=dataset)