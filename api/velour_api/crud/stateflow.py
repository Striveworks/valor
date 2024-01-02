from enum import Enum
from functools import wraps

from velour_api.crud.jobs import Job, generate_uuid, get_status_from_uuid
from velour_api.enums import JobStatus
from velour_api.exceptions import (
    DatasetAlreadyExistsError,
    DatasetDoesNotExistError,
    DatasetFinalizedError,
    DatasetNotFinalizedError,
    JobStateError,
    ModelAlreadyExistsError,
    ModelDoesNotExistError,
    ModelFinalizedError,
    ModelNotFinalizedError,
)


class StateflowNode(Enum):
    DATASET = "dataset"
    MODEL = "model"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"


class StateflowJob:
    """
    Stateflow helper object for managing job statuses.

    This object is designed to assist in managing the workflow of jobs and their statuses
    within the stateflow architecture. It interacts with the `Job` class and provides
    methods for setting and updating job statuses based on predefined states.

    The input must include one of the following sets to properly fetch a job:

    - Dataset Job: {dataset_name}
    - Model Job: {model_name}
    - Model Prediction Job: {dataset_name, model_name}
    - Evaluation Job: {evaluation_id} or {dataset_name, model_name, evaluation_id}

    Parameters
    ----------
    dataset_name : str or None
        Name of the dataset.
    model_name : str or None
        Name of the model.
    evaluation_id : int or None
        Unique identifier for the evaluation.
    start : JobStatus
        Initial status of the job.
    success : JobStatus
        Status to be set on successful completion.
    failure : JobStatus
        Status to be set on failure.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    evaluation_id : int
        Unique identifier for the evaluation.
    start : JobStatus
        Status to be set before execution.
    success : JobStatus
        Status to be set after a successful execution.
    failure : JobStatus
        Status to be set after a failed execution.
    uuid : str
        UUID for the StateflowJob.
    dataset_uuid : str
        UUID for the dataset-related job.
    model_uuid : str
        UUID for the model-related job.
    prediction_uuid : str
        UUID for the prediction-related job.
    evaluation_uuid : str
        UUID for the evaluation-related job.
    job : Job
        Job instance associated with the StateflowJob.
    node : StateflowNode
        Node type representing the context of the StateflowJob.

    Raises
    ------
    ValueError
        If the input parameters do not match any valid StateflowNode.

    Methods
    -------
    set_status(status: JobStatus, msg: str = "")
        Wraps `Job.set_status` to set the status of the associated job.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        evaluation_id: int,
        start: JobStatus,
        success: JobStatus,
        failure: JobStatus,
    ):
        # store input args
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.evaluation_id = evaluation_id
        self.start = start
        self.success = success
        self.failure = failure

        # generate uuids
        self.uuid = generate_uuid(
            dataset_name=dataset_name,
            model_name=model_name,
            evaluation_id=evaluation_id,
        )
        self.dataset_uuid = generate_uuid(dataset_name=dataset_name)
        self.model_uuid = generate_uuid(model_name=model_name)
        self.prediction_uuid = generate_uuid(
            dataset_name=dataset_name, model_name=model_name
        )
        self.evaluation_uuid = generate_uuid(
            dataset_name=dataset_name,
            model_name=model_name,
            evaluation_id=evaluation_id,
        )

        # create or get job
        self.job = Job.get(self.uuid)

        # get node
        if dataset_name and model_name and evaluation_id:
            self.node = StateflowNode.EVALUATION
        elif dataset_name and model_name and not evaluation_id:
            self.node = StateflowNode.PREDICTION
        elif dataset_name and not (model_name or evaluation_id):
            self.node = StateflowNode.DATASET
        elif model_name and not (dataset_name or evaluation_id):
            self.node = StateflowNode.MODEL
        else:
            raise ValueError("")

    def set_status(self, status: JobStatus, msg: str = ""):
        """
        Wraps `Job.set_status` to set the status of the associated job.
        """
        self.job.set_status(status, msg)


def _validate_transition(state: StateflowJob):
    """
    Validate edge-cases that require knowledge of the next transistion.
    """

    job = state.job
    node = state.node
    dataset_name = state.dataset_name
    model_name = state.model_name
    evaluation_id = state.evaluation_id
    dataset_uuid = state.dataset_uuid

    current_status = job.status

    # catch all errors from illegal transitions
    if state.start not in current_status.next():

        # attempt to create after finalization.
        if (
            state.start == JobStatus.CREATING
            and current_status == JobStatus.DONE
        ):
            if node == StateflowNode.DATASET:
                raise DatasetFinalizedError(dataset_name)
            elif node == StateflowNode.MODEL:
                raise ModelAlreadyExistsError(model_name)
            elif node == StateflowNode.PREDICTION:
                raise ModelFinalizedError(
                    dataset_name=dataset_name, model_name=model_name
                )
            elif node == StateflowNode.EVALUATION:
                raise JobStateError(
                    id=job.uuid,
                    msg=f"Evaluation {evaluation_id} already exists.",
                )

        # attempt to process before finalization
        if (
            current_status == JobStatus.CREATING
            and state.start == JobStatus.PROCESSING
        ):
            if node == StateflowNode.PREDICTION:
                raise ModelNotFinalizedError(
                    dataset_name=dataset_name, model_name=model_name
                )

        # attempt to create while deleting
        if (
            current_status == JobStatus.DELETING
            and state.start == JobStatus.CREATING
        ):
            if node == StateflowNode.DATASET:
                raise DatasetAlreadyExistsError(name=state.dataset_name)
            elif node == StateflowNode.MODEL:
                raise ModelAlreadyExistsError(name=state.model_name)

        # attempt to delete when does not exist
        if (
            current_status == JobStatus.PENDING
            and state.start == JobStatus.DELETING
        ):
            if node == StateflowNode.DATASET:
                raise DatasetDoesNotExistError(name=state.dataset_name)
            elif node == StateflowNode.MODEL:
                raise ModelDoesNotExistError(name=state.model_name)

        raise JobStateError(
            job.uuid,
            f"Requested transition from {current_status} to {state.start} is illegal.",
        )

    # catch un-finalized parents, this cannot be done before as predictions and evaluation use the same node.
    if state.start == JobStatus.PROCESSING:
        if get_status_from_uuid(dataset_uuid) == JobStatus.CREATING:
            raise DatasetNotFinalizedError(name=dataset_name)


def _validate_parents(state: StateflowJob):
    """
    Validate that parent jobs are finished.
    """

    job = state.job
    node = state.node
    dataset_name = state.dataset_name
    model_name = state.model_name
    dataset_uuid = state.dataset_uuid
    model_uuid = state.model_uuid
    prediction_uuid = state.prediction_uuid

    # validate parents of evaluations (dataset/groundtruths + predictions)
    if node == StateflowNode.EVALUATION:

        # dataset and groundtruths are still being created.
        if get_status_from_uuid(dataset_uuid) != JobStatus.DONE:
            raise DatasetNotFinalizedError(name=dataset_name)

        # model does not exist.
        elif get_status_from_uuid(model_uuid) == JobStatus.NONE:
            raise ModelDoesNotExistError(name=model_name)

        # model is still being created.
        elif get_status_from_uuid(model_uuid) == JobStatus.CREATING:
            raise ModelNotFinalizedError(
                dataset_name=dataset_name, model_name=model_name
            )

        # predictions are still being created.
        elif get_status_from_uuid(prediction_uuid) != JobStatus.DONE:
            raise ModelNotFinalizedError(
                dataset_name=dataset_name, model_name=model_name
            )

        # register job as child of parents
        Job.get(prediction_uuid).register_child(job.uuid)
        Job.get(dataset_uuid).register_child(job.uuid)

    # validate parents of predictions (dataset + model)
    elif node == StateflowNode.PREDICTION:

        # dataset has not been created.
        if get_status_from_uuid(dataset_uuid) not in [
            JobStatus.CREATING,
            JobStatus.DONE,
        ]:
            raise DatasetDoesNotExistError(name=dataset_name)

        # model has not been created.
        elif get_status_from_uuid(model_uuid) not in [
            JobStatus.CREATING,
            JobStatus.DONE,
        ]:
            raise ModelDoesNotExistError(name=model_name)

        # register job as child of parents
        Job.get(model_uuid).register_child(job.uuid)

    # no parent nodes
    elif node == StateflowNode.DATASET:
        pass
    elif node == StateflowNode.MODEL:
        pass
    else:
        raise ValueError("Received invalid input.")


def _validate_children(state: StateflowJob):
    """
    Validate the children of a job are finished (if they exist).
    """

    job = state.job

    # edge case
    if state.start == JobStatus.DELETING:
        return

    def _recursive_child_search(job: Job):
        for uuid in job.children:
            status = get_status_from_uuid(uuid=uuid)
            # throw exception if child is not in a stable state
            if status not in [
                JobStatus.NONE,
                JobStatus.DONE,
                JobStatus.FAILED,
            ]:
                raise JobStateError(
                    job.uuid,
                    f"Job blocked by child task with uuid `{uuid}` and status `{get_status_from_uuid(uuid=uuid).value}`",
                )
            elif status == JobStatus.DONE:
                _recursive_child_search(Job.get(uuid))

    _recursive_child_search(job)


def _parse_kwargs(kwargs: dict) -> tuple:
    """
    Parses kwargs into {dataset_name, model_name, evaluation_id}.
    """
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
    return (dataset_name, model_name, evaluation_id)


def generate_stateflow_decorator(
    start: JobStatus = JobStatus.PROCESSING,
    success: JobStatus = JobStatus.DONE,
    failure: JobStatus = JobStatus.FAILED,
    on_start: callable = lambda state, msg="": state.set_status(
        state.start, msg
    ),
    on_success: callable = lambda state, msg="": state.set_status(
        state.success, msg
    ),
    on_failure: callable = lambda state, msg="": state.set_status(
        state.failure, msg
    ),
):
    """
    Decorator generator function for creating Stateflow decorators.

    This function generates decorators for specific states in the Stateflow.
    The generated decorators can be used to wrap functions or methods, allowing for
    easy management of job statuses based on predefined states.

    Parameters
    ----------
    start : JobStatus, default=JobStatus.PROCESSING
        Initial status of the job.
    success : JobStatus, default=JobStatus.DONE
        Status to be set on successful completion.
    failure : JobStatus, default=JobStatus.FAILED
        Status to be set on failure.
    on_start : callable, optional
        Function to be executed when transitioning to the start state.
    on_success : callable, optional
        Function to be executed when transitioning to the success state.
    on_failure : callable, optional
        Function to be executed when transitioning to the failure state.

    Returns
    -------
    Callable
        A decorator that can be used to wrap functions or methods and manage job
        statuses based on predefined states.

    Notes
    -----
    The `on_start`, `on_success`, and `on_failure` functions are executed when
    transitioning to the corresponding states and can be customized based on
    specific requirements.
    """

    def decorator(fn: callable) -> callable:
        @wraps(fn)
        def wrapper(*args, precheck: bool = False, **kwargs):
            if len(args) > 0:
                raise ValueError(
                    "Stateflow decorator can only recognize explicit arguments (kwargs)."
                )

            dataset_name, model_name, evaluation_id = _parse_kwargs(kwargs)

            # validate job state
            state = StateflowJob(
                dataset_name=dataset_name,
                model_name=model_name,
                evaluation_id=evaluation_id,
                start=start,
                success=success,
                failure=failure,
            )
            _validate_transition(state)
            _validate_parents(state)
            _validate_children(state)

            # If precheck is defined as True then return.
            # This exists for background tasks that just need
            # to run the validation step.
            if precheck:
                return

            # wrapped function execution
            on_start(state)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                on_failure(state, str(e))
                raise e
            on_success(state)

            return result

        return wrapper

    return decorator


def _finalize_success(state: StateflowJob, msg: str = ""):
    """
    Since model and predictions are handled seperately this helper function is necessary to ensure
    that a finalization call over a prediction set will also finalize the model itself.
    """
    if state.node == StateflowNode.PREDICTION:
        if get_status_from_uuid(state.model_uuid) == JobStatus.CREATING:
            Job.get(state.model_uuid).set_status(state.success, msg)
    state.set_status(state.success, msg)


# stateflow decorator definitions
create = generate_stateflow_decorator(
    start=JobStatus.CREATING,
    success=JobStatus.CREATING,
    failure=JobStatus.CREATING,
)
finalize = generate_stateflow_decorator(
    start=JobStatus.CREATING,
    success=JobStatus.DONE,
    failure=JobStatus.CREATING,
    on_success=_finalize_success,
)
evaluate = generate_stateflow_decorator(
    start=JobStatus.PROCESSING,
    success=JobStatus.DONE,
)
delete = generate_stateflow_decorator(
    start=JobStatus.DELETING,
    on_success=lambda state, msg="": state.job.delete(),
)
