import re
from functools import wraps
from pydantic import BaseModel

from velour_api import logger, schemas
from velour_api.crud import jobs
from velour_api.crud.jobs import (
    get_status_from_uuid,
    get_status_from_names,
    generate_uuid,
    Job,
)
from velour_api.enums import JobStatus
from velour_api.exceptions import (
    JobStateError, 
    DatasetDoesNotExistError,
    DatasetNotFinalizedError,
    DatasetFinalizedError,
    ModelDoesNotExistError,
    ModelAlreadyExistsError,
    ModelNotFinalizedError, 
    ModelFinalizedError,
)


class StateTransition(BaseModel):
    start: JobStatus = JobStatus.PROCESSING
    success: JobStatus = JobStatus.DONE
    failure: JobStatus = JobStatus.FAILED


def _validate_parents(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
):
    """
    Safely retrieves or creates a Redis job through parental checks.
    """

    dataset_uuid = generate_uuid(dataset_name=dataset_name)
    model_uuid = generate_uuid(model_name=model_name)
    inference_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name)
    evaluation_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name, evaluation_id=evaluation_id)
    
    # validate parents of evaluations (groundtruths + predictions)
    if evaluation_id and dataset_name and model_name:
        
        # dataset and groundtruths are still being created.
        if get_status_from_uuid(dataset_uuid) != JobStatus.DONE:
            raise DatasetNotFinalizedError(name=dataset_name)
        
        # model is still being created.
        elif get_status_from_uuid(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        
        # model predictions are still being created.
        elif get_status_from_uuid(inference_uuid) != JobStatus.DONE:
            raise ModelNotFinalizedError(dataset_name=dataset_name, model_name=model_name)
        
        job = Job.get(evaluation_uuid)
        Job.get(inference_uuid).register_child(job.uuid)
        Job.get(dataset_uuid).register_child(job.uuid)

    # validate parents of predictions (dataset + model)
    elif dataset_name and model_name:

        # dataset is not finalized or being created.
        if get_status_from_uuid(dataset_uuid) not in [JobStatus.CREATING, JobStatus.DONE]:
            raise DatasetDoesNotExistError(name=dataset_name)
        
        # model has not been created.
        elif get_status_from_uuid(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        
        job = Job.get(inference_uuid)
        Job.get(model_uuid).register_child(job.uuid)    

    # no parent nodes
    elif dataset_name:
        job = Job.get(dataset_uuid)
    elif model_name:
        job = Job.get(model_uuid)
    else:
        raise ValueError(f"Received invalid input.")
    
    return job


def _validate_children(job: Job):
    """
    Validate the children of a Job (if they exist).
    """
    def _recursive_child_search(job: Job):
        # Check status of child jobs
        for uuid in job.children:
            status = get_status_from_uuid(uuid=uuid)
            if status not in [JobStatus.NONE, JobStatus.DONE, JobStatus.FAILED]:
                raise JobStateError(job.uuid, f"Job blocked by child task with uuid `{uuid}` and status `{get_status_from_uuid(uuid=uuid).value}`")
            elif status == JobStatus.DONE:
                _recursive_child_search(Job.get(uuid))
    _recursive_child_search(job)
    

def _validate_transition(
    job: Job,
    transitions: StateTransition,
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
):
    """
    Validate edge-cases that require knowledge of the next transistion.
    """
    
    dataset_uuid = generate_uuid(dataset_name=dataset_name)
    model_uuid = generate_uuid(model_name=model_name)
    inference_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name)
    evaluation_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name, evaluation_id=evaluation_id)

    # validate state transition
    current_status = job.status
    if transitions.start not in current_status.next():
        if transitions.start == JobStatus.CREATING and current_status == JobStatus.DONE:
            if dataset_name and not (model_name or evaluation_id):
                raise DatasetFinalizedError(dataset_name)
            elif model_name and not (dataset_name or evaluation_id):
                raise ModelAlreadyExistsError(model_name)
            elif model_name and dataset_name and not evaluation_id:
                raise ModelFinalizedError(dataset_name=dataset_name, model_name=model_name)
            elif model_name and dataset_name and evaluation_id:
                raise JobStateError(id=job.uuid, msg=f"Evaluation {evaluation_id} already exists.")
    if transitions.start == JobStatus.PROCESSING:
        if get_status_from_uuid(dataset_uuid) == JobStatus.CREATING:
            raise DatasetNotFinalizedError(name=dataset_name)
        elif get_status_from_uuid(inference_uuid) == JobStatus.CREATING:
            raise ModelNotFinalizedError(dataset_name=dataset_name, model_name=model_name)
        

def get_job(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> Job:
    """
    Safely get or create a Job.
    """
    job = _validate_parents(
        dataset_name=dataset_name,
        model_name=model_name,
        evaluation_id=evaluation_id,
    )
    _validate_children(job)
    return job


def get_status(
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> JobStatus:
    """
    Get status of a Job.
    """
    return get_status_from_names(
        dataset_name=dataset_name,
        model_name=model_name,
        evaluation_id=evaluation_id,
    )


def generate_stateflow_decorator(
    transitions: StateTransition = StateTransition(),
    on_start: callable = lambda job, transitions, msg="" : job.set_status(transitions.start, msg),
    on_success: callable = lambda job, transitions, msg="" : job.set_status(transitions.success, msg),
    on_failure: callable = lambda job, transitions, msg="" : job.set_status(transitions.failure, msg),
):
    """
    Decorator generator function.
    """
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
                evaluation_id = kwargs["job_request"].id
                if "job_id" in kwargs:
                    evaluation_id = kwargs["job_id"]
            elif "dataset_name" in kwargs and "model_name" in kwargs:
                dataset_name = kwargs["dataset_name"]
                model_name = kwargs["model_name"]
            elif "dataset_name" in kwargs:
                dataset_name = kwargs["dataset_name"]
            elif "model_name" in kwargs:
                model_name = kwargs["model_name"]
            else:
                raise ValueError("did not receive right values")

            job = _validate_parents(
                dataset_name=dataset_name,
                model_name=model_name,
                evaluation_id=evaluation_id,
            )

            if transitions.start != JobStatus.DELETING:
                _validate_children(job)

            _validate_transition(
                job=job,
                transitions=transitions,
                dataset_name=dataset_name,
                model_name=model_name,
                evaluation_id=evaluation_id
            )

            on_start(job, transitions)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                on_failure(job, transitions, str(e))
                raise e
            on_success(job, transitions)
            return result
        return wrapper
    return decorator


# stateflow decorators
create = generate_stateflow_decorator(
    transitions=StateTransition(
        start=JobStatus.CREATING,
        success=JobStatus.CREATING,
    ),
)
finalize = generate_stateflow_decorator(
    transitions=StateTransition(
        start=JobStatus.CREATING,
        success=JobStatus.DONE,
    ),
)
evaluate = generate_stateflow_decorator(
    transitions=StateTransition(
        start=JobStatus.PROCESSING,
        success=JobStatus.DONE,
    ),
)
delete = generate_stateflow_decorator(
    transitions=StateTransition(
        start=JobStatus.DELETING,
    ),
    on_success=lambda job, transitions, msg="" : job.delete(),
)

