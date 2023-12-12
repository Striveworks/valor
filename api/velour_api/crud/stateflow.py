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


class JobValidator:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        evaluation_id: int,
        transitions: StateTransition,
    ):
        # store input args
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.evaluation_id = evaluation_id
        self.transitions = transitions

        # generate uuids
        self.uuid = generate_uuid(
            dataset_name=dataset_name,
            model_name=model_name,
            evaluation_id=evaluation_id,
        )
        self.dataset_uuid = generate_uuid(dataset_name=dataset_name)
        self.model_uuid = generate_uuid(model_name=model_name)
        self.prediction_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name)
        self.evaluation_uuid = generate_uuid(dataset_name=dataset_name, model_name=model_name, evaluation_id=evaluation_id)

        # create or get job
        self.job = Job.get(self.uuid)


def _validate_transition(validator: JobValidator):
    """
    Validate edge-cases that require knowledge of the next transistion.
    """

    job = validator.job
    transitions = validator.transitions
    dataset_name = validator.dataset_name
    model_name = validator.model_name
    evaluation_id = validator.evaluation_id
    dataset_uuid =  validator.dataset_uuid
    model_uuid = validator.model_uuid
    prediction_uuid = validator.prediction_uuid
    evaluation_uuid = validator.evaluation_uuid

    current_status = job.status

    # catch all errors from illegal transitions
    if transitions.start not in current_status.next():

        # attempt to create after finalization.
        if transitions.start == JobStatus.CREATING and current_status == JobStatus.DONE:
            if dataset_name and not (model_name or evaluation_id):
                raise DatasetFinalizedError(dataset_name)
            elif model_name and not (dataset_name or evaluation_id):
                raise ModelAlreadyExistsError(model_name)
            elif model_name and dataset_name and not evaluation_id:
                raise ModelFinalizedError(dataset_name=dataset_name, model_name=model_name)
            elif model_name and dataset_name and evaluation_id:
                raise JobStateError(id=job.uuid, msg=f"Evaluation {evaluation_id} already exists.")
            
        # attempt to process before finalization
        if transitions.start == JobStatus.PROCESSING and current_status == JobStatus.CREATING:
            if model_name and dataset_name and not evaluation_id:
                raise ModelNotFinalizedError(dataset_name=dataset_name, model_name=model_name)
        
        raise JobStateError(job.uuid, f"Requested transition from {current_status} to {transitions.start} is illegal.")
            
    # catch un-finalized parents, this cannot be done before as predictions and evaluation use the same node.
    if transitions.start == JobStatus.PROCESSING:
        if get_status_from_uuid(dataset_uuid) == JobStatus.CREATING:
            raise DatasetNotFinalizedError(name=dataset_name)


def _validate_parents(validator: JobValidator):
    """
    Validate that parent jobs are finished.
    """

    job = validator.job
    transitions = validator.transitions
    dataset_name = validator.dataset_name
    model_name = validator.model_name
    evaluation_id = validator.evaluation_id
    dataset_uuid =  validator.dataset_uuid
    model_uuid = validator.model_uuid
    prediction_uuid = validator.prediction_uuid
    evaluation_uuid = validator.evaluation_uuid
    
    # validate parents of evaluations (dataset/groundtruths + predictions)
    if evaluation_id and dataset_name and model_name:
        
        # dataset and groundtruths are still being created.
        if get_status_from_uuid(dataset_uuid) != JobStatus.DONE:
            raise DatasetNotFinalizedError(name=dataset_name)
        
        # model is still being created.
        elif get_status_from_uuid(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        
        # predictions are still being created.
        elif get_status_from_uuid(prediction_uuid) != JobStatus.DONE:
            raise ModelNotFinalizedError(dataset_name=dataset_name, model_name=model_name)
        
        # register job as child of parents
        Job.get(prediction_uuid).register_child(job.uuid)
        Job.get(dataset_uuid).register_child(job.uuid)

    # validate parents of predictions (dataset + model)
    elif dataset_name and model_name:
        
        # dataset is not finalized or being created.
        if get_status_from_uuid(dataset_uuid) not in [JobStatus.CREATING, JobStatus.DONE]:
            raise DatasetDoesNotExistError(name=dataset_name)

        # model has not been created.
        elif get_status_from_uuid(model_uuid) != JobStatus.DONE:
            raise ModelDoesNotExistError(name=model_name)
        
        # register job as child of parents
        Job.get(model_uuid).register_child(job.uuid)

    # no parent nodes
    elif dataset_name:
        pass
    elif model_name:
        pass
    else:
        raise ValueError(f"Received invalid input.")


def _validate_children(validator: JobValidator):
    """
    Validate the children of a job are finished (if they exist).
    """

    job = validator.job
    transitions = validator.transitions

    # edge case
    if transitions.start == JobStatus.DELETING:
        return
    
    def _recursive_child_search(job: Job):
        for uuid in job.children:
            status = get_status_from_uuid(uuid=uuid)
            # throw exception if child is not in a stable state
            if status not in [JobStatus.NONE, JobStatus.DONE, JobStatus.FAILED]:
                raise JobStateError(job.uuid, f"Job blocked by child task with uuid `{uuid}` and status `{get_status_from_uuid(uuid=uuid).value}`")
            elif status == JobStatus.DONE:
                _recursive_child_search(Job.get(uuid))
    _recursive_child_search(job)


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

            validator = JobValidator(
                dataset_name=dataset_name,
                model_name=model_name,
                evaluation_id=evaluation_id,
                transitions=transitions,
            )
            _validate_transition(validator)
            _validate_parents(validator)
            _validate_children(validator)

            job = validator.job

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


# stateflow decorator definitions
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

