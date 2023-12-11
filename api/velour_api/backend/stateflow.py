from velour_api import logger, schemas
from velour_api.backend.jobs import get_stateflow, set_stateflow
from velour_api.enums import JobStatus, State


def _update_backend_state(
    *,
    status: State,
    dataset_name: str | None = None,
    model_name: str | None = None,
):
    # retrieve from redis
    stateflow = get_stateflow()

    # update stateflow object
    if dataset_name and model_name:
        stateflow.set_inference_status(
            dataset_name=dataset_name,
            model_name=model_name,
            status=status,
        )
    elif dataset_name:
        stateflow.set_dataset_status(
            dataset_name=dataset_name,
            status=status,
        )
    elif model_name:
        stateflow.set_model_status(
            model_name=model_name,
            status=status,
        )
    else:
        # do nothing
        return

    # commit to redis
    set_stateflow(stateflow)


def _update_job_state(
    *,
    dataset_name: str,
    model_name: str,
    job_id: int,
    status: JobStatus,
):
    # retrieve from redis
    stateflow = get_stateflow()
    if stateflow.get_job_status(job_id) == JobStatus.DONE:
        return

    # update stateflow object
    stateflow.set_job_status(
        dataset_name=dataset_name,
        model_name=model_name,
        job_id=job_id,
        status=status,
    )

    # commit to redis
    set_stateflow(stateflow)


def _remove_backend_state(
    *,
    dataset_name: str | None = None,
    model_name: str | None = None,
):
    # retrieve from redis
    stateflow = get_stateflow()

    # update stateflow object
    if dataset_name and model_name:
        stateflow.remove_inference(
            dataset_name=dataset_name, model_name=model_name
        )
    elif dataset_name:
        stateflow.remove_dataset(dataset_name)
    elif model_name:
        stateflow.remove_model(model_name)
    else:
        # do nothing
        return

    # commit to redis
    set_stateflow(stateflow)


def create(fn: callable) -> callable:
    """
    Set the state of a dataset and/or model object to State.CREATE.

    Parameters
    ----------
    fn : callable
        An input function to wrap around. Sets the state of any datasets or models mentioned in the function's kwargs.

    Raises
    ------
    RuntimeError
        If input args aren't explicitly defined.
    """

    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError("input arguments should be explicitly defined")

        # unpack args
        dataset_name = None
        model_name = None
        state = State.CREATE

        # create grouping
        if "dataset" in kwargs:
            if isinstance(kwargs["dataset"], schemas.Dataset):
                dataset_name = kwargs["dataset"].name
                model_name = None
                state = State.NONE
        elif "model" in kwargs:
            if isinstance(kwargs["model"], schemas.Dataset):
                dataset_name = None
                model_name = kwargs["model"].name
                state = State.NONE
        # create annotations
        elif "groundtruth" in kwargs:
            if isinstance(kwargs["groundtruth"], schemas.GroundTruth):
                dataset_name = kwargs["groundtruth"].datum.dataset
                model_name = None
        elif "prediction" in kwargs:
            if isinstance(kwargs["prediction"], schemas.Prediction):
                dataset_name = kwargs["prediction"].datum.dataset
                model_name = kwargs["prediction"].model

        _update_backend_state(
            status=state,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    """
    Set the state of a dataset and/or model object to State.Finalize.

    Parameters
    ----------
    fn : callable
        An input function to wrap around. Sets the state of any datasets or models mentioned in the function's kwargs.

    Raises
    ------
    RuntimeError
        If input args aren't explicitly defined.
    """

    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 3:
            raise RuntimeError

        # unpack dataset
        dataset_name = None
        if "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]

        # unpack model
        model_name = None
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]

        _update_backend_state(
            status=State.READY,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    """
    Set the state of a dataset and/or model object to State.EVALUATE.

    Parameters
    ----------
    fn : callable
        An input function to wrap around. Sets the state of any datasets or models mentioned in the function's kwargs.

    Raises
    ------
    RuntimeError
        If input args aren't explicitly defined.
    """

    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None
        if "job_request" in kwargs:
            dataset_name = kwargs["job_request"].dataset
            model_name = kwargs["job_request"].model

        _update_backend_state(
            status=State.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        try:
            results = fn(*args, **kwargs)
        except Exception as e:
            _update_backend_state(
                status=State.READY,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            logger.debug(f"Evaluation request failed. Exception: {str(e)}")
            raise e

        if hasattr(results, "job_id"):
            _update_job_state(
                dataset_name=dataset_name,
                model_name=model_name,
                job_id=results.job_id,
                status=JobStatus.PENDING,
            )

        return results

    return wrapper


def computation(fn: callable) -> callable:
    """
    Update the state of an EvaluationJob.

    Parameters
    ----------
    fn : callable
        An input function to wrap around.

    Raises
    ------
    RuntimeError
        If input args aren't explicitly defined.
    ValueError
        If the job request is defined, but isn't the right type.
        If the job request isn't defined.
        If the job is missing an ID.
    """

    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack job_request
        if "job_request" in kwargs:
            if isinstance(kwargs["job_request"], schemas.EvaluationJob):
                dataset_name = kwargs["job_request"].dataset
                model_name = kwargs["job_request"].model
            else:
                raise ValueError(
                    "job_request object must be of type `schemas.EvaluationJob`"
                )
        else:
            raise ValueError(
                "missing job_request which should be an evaluation request type (e.g. `schemas.EvaluationJob`)"
            )

        # unpack job_id
        if "job_id" in kwargs:
            job_id = kwargs["job_id"]
        else:
            raise ValueError("missing job_id")

        if get_stateflow().get_job_status(job_id) == JobStatus.DONE:
            _update_backend_state(
                status=State.READY,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            return

        # set up stateflow for pre-computation
        _update_backend_state(
            status=State.EVALUATE,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        _update_job_state(
            dataset_name=dataset_name,
            model_name=model_name,
            job_id=job_id,
            status=JobStatus.PROCESSING,
        )

        try:
            # compute
            result = fn(*args, **kwargs)
        except Exception as e:
            # failed
            _update_job_state(
                dataset_name=dataset_name,
                model_name=model_name,
                job_id=job_id,
                status=JobStatus.FAILED,
            )
            logger.debug(f"job with id `{job_id}` failed. Exception: {str(e)}")
            raise e
        else:
            # success
            _update_job_state(
                dataset_name=dataset_name,
                model_name=model_name,
                job_id=job_id,
                status=JobStatus.DONE,
            )
        finally:
            # return to ready state
            _update_backend_state(
                status=State.READY,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        return result

    return wrapper


def delete(fn: callable) -> callable:
    """
    Set the state of a dataset and/or model object to State.DELETE.

    Parameters
    ----------
    fn : callable
        An input function to wrap around. Sets the state of any datasets or models mentioned in the function's kwargs.

    Raises
    ------
    RuntimeError
        If input args aren't explicitly defined.
    """

    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 3:
            raise RuntimeError

        # unpack dataset
        dataset_name = None
        if "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]

        # unpack model
        model_name = None
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]

        _update_backend_state(
            status=State.DELETE,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        try:
            result = fn(*args, **kwargs)
        finally:
            _remove_backend_state(
                dataset_name=dataset_name, model_name=model_name
            )

        return result

    return wrapper
