from time import perf_counter

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

    # check if job has already completed (if it exists)
    if (
        stateflow.get_job_status(dataset_name, model_name, job_id)
        == JobStatus.DONE
    ):
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
    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None
        if "request_info" in kwargs:
            dataset_name = kwargs["request_info"].settings.dataset
            model_name = kwargs["request_info"].settings.model

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
    def wrapper(*args, **kwargs):
        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack request_info
        if "request_info" in kwargs:
            if hasattr(kwargs["request_info"], "settings"):
                if isinstance(
                    kwargs["request_info"].settings, schemas.EvaluationSettings
                ):
                    dataset_name = kwargs["request_info"].settings.dataset
                    model_name = kwargs["request_info"].settings.model
            else:
                raise ValueError(
                    "request_info object must contain an attribute named `settings` of type `schemas.EvaluationSettings`"
                )
        else:
            raise ValueError(
                "missing request_info which should be an evaluation request type (e.g. `schemas.APRequest`)"
            )

        # unpack job_id
        if "job_id" in kwargs:
            job_id = kwargs["job_id"]
        else:
            raise ValueError("missing job_id")

        # check if job has already successfully ran
        if (
            get_stateflow().get_job_status(dataset_name, model_name, job_id)
            == JobStatus.DONE
        ):
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


def debug_timer(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        logger.debug(f"starting method {fn}")
        start = perf_counter()
        result = fn(*args, **kwargs)
        logger.debug(
            f"method {fn} finished in {perf_counter() - start} seconds"
        )
        return result

    return wrapper
