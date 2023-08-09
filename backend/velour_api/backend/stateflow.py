from time import perf_counter

from velour_api import logger, schemas
from velour_api.backend.jobs import get_backend_state, set_backend_state
from velour_api.enums import Stateflow


def _update_backend_status(
    status: Stateflow,
    dataset_name: str,
    model_name: str | None = None,
):
    # get current status
    current_status = get_backend_state()

    # update status
    if model_name:
        current_status.set_inference_status(
            status=status,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    else:
        current_status.set_dataset_status(
            dataset_name=dataset_name, status=status
        )

    set_backend_state(current_status)


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None

        # create grouping
        if "dataset" in kwargs:
            if isinstance(kwargs["dataset"], schemas.Dataset):
                dataset_name = kwargs["dataset"].name
                model_name = None
        elif "model" in kwargs:  # @TODO: This is option does nothing currently
            if isinstance(kwargs["model"], schemas.Dataset):
                dataset_name = None
                model_name = kwargs["model"].name

        # create annotations
        elif "groundtruth" in kwargs:
            if isinstance(kwargs["groundtruth"], schemas.GroundTruth):
                dataset_name = kwargs["groundtruth"].datum.dataset
                model_name = None
        elif "prediction" in kwargs:
            if isinstance(kwargs["prediction"], schemas.Prediction):
                dataset_name = kwargs["prediction"].datum.dataset
                model_name = kwargs["prediction"].model

        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.CREATE,
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

        # enter ready state
        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.READY,
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
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        # put model / dataset in evaluation state
        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.EVALUATE,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        return fn(*args, **kwargs)

    return wrapper


def computation(fn: callable) -> callable:
    def wrapper(*args, **kwargs):

        # input args should be explicitly defined
        if len(args) != 0 and len(kwargs) != 2:
            raise RuntimeError

        # unpack args
        dataset_name = None
        model_name = None
        if "request_info" in kwargs:
            if isinstance(kwargs["request_info"], schemas.ClfMetricsRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model
            elif isinstance(kwargs["request_info"], schemas.APRequest):
                dataset_name = kwargs["request_info"].settings.dataset
                model_name = kwargs["request_info"].settings.model

        # start eval computation
        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.EVALUATE,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        result = fn(*args, **kwargs)

        # end eval computation
        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.READY,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        return result

    return wrapper


# @TODO: Need to find better solution than just catching error when deleting model that had predictions
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

        if dataset_name is not None:
            _update_backend_status(
                status=Stateflow.DELETE,
                dataset_name=dataset_name,
                model_name=model_name,
            )

        result = fn(*args, **kwargs)

        if dataset_name is not None:
            status = get_backend_state()
            if model_name is not None:
                status.remove_model(model_name)
            else:
                status.remove_dataset(dataset_name)
            set_backend_state(status)

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
