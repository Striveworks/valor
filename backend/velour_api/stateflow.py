from functools import wraps
from warnings import warn

from velour_api import exceptions
from velour_api.enums import TableStatus
from velour_api.jobs import get_status, set_status
from velour_api.schemas import (
    APRequest,
    ClfMetricsRequest,
    Dataset,
    DatasetStatus,
    GroundTruthClassificationsCreate,
    GroundTruthDetectionsCreate,
    GroundTruthSegmentationsCreate,
    Model,
    PredictedClassificationsCreate,
    PredictedDetectionsCreate,
    PredictedSegmentationsCreate,
)


def get_dataset_status(dataset_name: str):
    status = get_status()
    if dataset_name not in status:
        raise exceptions.DatasetDoesNotExistError(dataset_name)
    elif not status.datasets[dataset_name].valid():
        raise exceptions.StateflowError(
            f"'{dataset_name}' is in a invalid state."
        )
    return status.datasets[dataset_name].status


def get_inference_status(dataset_name: str, model_name: str):
    status = get_status()
    if dataset_name not in status.datasets:
        raise exceptions.DatasetDoesNotExistError(dataset_name)
    elif not status.datasets[dataset_name].valid():
        raise exceptions.StateflowError(
            f"'{dataset_name}' is in a invalid state."
        )
    elif model_name not in status.datasets[dataset_name].models:
        raise exceptions.InferenceDoesNotExistError(dataset_name, model_name)
    return status.datasets[dataset_name].models[model_name]


def set_dataset_status(dataset_name: str, state: TableStatus):
    status = get_status()

    # Check if dataset needs to be created
    if dataset_name not in status.datasets:

        # Check if new state is for creation
        if state != TableStatus.CREATE:
            raise exceptions.DatasetDoesNotExistError(dataset_name)

        # Create new dataset
        status.datasets[dataset_name] = DatasetStatus(
            status=state,
            models={},
        )

    # Dataset exists
    else:

        # Check if new state is valid
        if state not in status.datasets[dataset_name].status.next():
            raise exceptions.InvalidStateTransitionError(
                status.datasets[dataset_name].status, state
            )

        # Check if inferences (if they exist) allow the next state
        if state not in status.datasets[dataset_name].next():
            raise exceptions.StateflowError(
                f"State transition blocked by existing inferences. {status.datasets[dataset_name].models.keys()} {status.datasets[dataset_name].next()}"
            )

        # Update the state
        if state == TableStatus.DELETE:
            for model_name in status.datasets[dataset_name].models:
                set_inference_status(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    state=TableStatus.DELETE,
                )
        status.datasets[dataset_name].status = state

    set_status(status)


def set_inference_status(
    dataset_name: str, model_name: str, state: TableStatus
):
    status = get_status()

    # Check if dataset exists
    if dataset_name not in status.datasets:
        raise exceptions.DatasetDoesNotExistError(dataset_name)

    # Check if dataset is in evaluation state
    if status.datasets[dataset_name].status != TableStatus.EVALUATE:
        if status.datasets[dataset_name].status == TableStatus.READY:
            set_dataset_status(dataset_name, TableStatus.EVALUATE)
        else:
            raise exceptions.StateflowError(
                f"dataset '{dataset_name}' is not in evaluation state."
            )

    # Check if inference does not exist
    if model_name not in status.datasets[dataset_name].models:

        # Check that the state is for creation
        if state != TableStatus.CREATE:
            if state == TableStatus.READY:
                warn("Model inferences empty.", RuntimeWarning)
            else:
                raise exceptions.InferenceDoesNotExistError(
                    dataset_name=dataset_name, model_name=model_name
                )

        # Create new inference
        status.datasets[dataset_name].models[model_name] = state

    # Inference exists
    else:

        # Check if new state is valid
        if (
            state
            not in status.datasets[dataset_name].models[model_name].next()
        ):
            raise exceptions.StateflowError(
                f"'{state}' is not a valid next state."
            )

        # Update the state
        status.datasets[dataset_name].models[model_name] = state

    set_status(status)


def remove_dataset(dataset_name: str):
    status = get_status()
    if dataset_name in status.datasets:
        if status.datasets[dataset_name].status == TableStatus.DELETE:
            del status.datasets[dataset_name]
            set_status(status)
        else:
            raise exceptions.StateflowError(
                "Attempted to delete dataset that was not in 'DELETE' state."
            )
    else:
        raise exceptions.DatasetDoesNotExistError(dataset_name)


def remove_model(model_name: str):
    status = get_status()
    for dataset_name in status.datasets:
        if model_name in status.datasets[dataset_name].models:
            if (
                status.datasets[dataset_name].models[model_name]
                == TableStatus.DELETE
            ):
                del status.datasets[dataset_name].models[model_name]
                set_status(status)
            else:
                raise exceptions.StateflowError(
                    f"Attempted to delete inference ({dataset_name}, {model_name}) that was not in 'DELETE' state."
                )
        if TableStatus.READY in status.datasets[dataset_name].next():
            set_dataset_status(dataset_name, TableStatus.READY)


def _create_dataset(dataset: Dataset):
    assert isinstance(dataset, Dataset)
    set_dataset_status(dataset_name=dataset.name, state=TableStatus.CREATE)


def _create_groundtruth(data):
    assert isinstance(
        data,
        (
            GroundTruthClassificationsCreate,
            GroundTruthDetectionsCreate,
            GroundTruthSegmentationsCreate,
        ),
    )
    set_dataset_status(
        dataset_name=data.dataset_name, state=TableStatus.CREATE
    )


def _create_prediction(data):
    assert isinstance(
        data,
        (
            PredictedClassificationsCreate,
            PredictedDetectionsCreate,
            PredictedSegmentationsCreate,
        ),
    )
    set_dataset_status(
        dataset_name=data.dataset_name,
        state=TableStatus.EVALUATE,
    )
    set_inference_status(
        model_name=data.model_name,
        dataset_name=data.dataset_name,
        state=TableStatus.CREATE,
    )


def _finalize_inference(model_name: str, dataset_name: str):
    assert isinstance(model_name, str)
    assert isinstance(dataset_name, str)
    set_inference_status(
        model_name=model_name,
        dataset_name=dataset_name,
        state=TableStatus.READY,
    )
    set_dataset_status(dataset_name=dataset_name, state=TableStatus.READY)


def _finalize_dataset(dataset_name: str):
    assert isinstance(dataset_name, str)
    set_dataset_status(dataset_name=dataset_name, state=TableStatus.READY)


def _evaluate_inference(request_info):
    assert isinstance(request_info, (APRequest, ClfMetricsRequest))
    set_dataset_status(
        dataset_name=request_info.settings.dataset_name,
        state=TableStatus.EVALUATE,
    )
    set_inference_status(
        model_name=request_info.settings.model_name,
        dataset_name=request_info.settings.dataset_name,
        state=TableStatus.EVALUATE,
    )


def _evaluate_inference_finished(request_info):
    assert isinstance(request_info, (APRequest, ClfMetricsRequest))
    set_inference_status(
        model_name=request_info.settings.model_name,
        dataset_name=request_info.settings.dataset_name,
        state=TableStatus.READY,
    )
    set_dataset_status(
        dataset_name=request_info.settings.dataset_name,
        state=TableStatus.READY,
    )


def _delete_dataset(dataset_name: str):
    set_dataset_status(dataset_name=dataset_name, state=TableStatus.DELETE)


def _delete_model(model_name: str):
    status = get_status()
    for dataset_name in status.datasets:
        if model_name in status.datasets[dataset_name].models:
            set_inference_status(
                model_name=model_name,
                dataset_name=dataset_name,
                state=TableStatus.DELETE,
            )


def _dereference_dataset(dataset_name: str):
    assert isinstance(dataset_name, str)
    remove_dataset(dataset_name)


def _dereference_model(model_name: str):
    assert isinstance(model_name, str)
    remove_model(model_name=model_name)


def create_dataset(fn: callable) -> callable:
    """Stateflow Create Dataset Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        if len(args) < 2:
            if "dataset" in kwargs:
                _create_dataset(kwargs["dataset"])
            elif "data" in kwargs:
                _create_groundtruth(kwargs["data"])
        elif len(args) == 2:
            if isinstance(args[1], Dataset):
                _create_dataset(args[1])
            else:
                _create_groundtruth(args[1])

        return fn(*args, **kwargs)

    return wrapper


def create_model(fn: callable) -> callable:
    """Stateflow Create Model Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        if len(args) < 2:
            if "model" in kwargs:
                pass
            elif "data" in kwargs:
                _create_prediction(kwargs["data"])
        elif len(args) == 2:
            if isinstance(args[1], Model):
                pass
            else:
                _create_prediction(args[1])

        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    """Stateflow Finalize Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        # Check if dataset_name exists
        dataset_name = None
        if "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        elif len(args) >= 2:
            dataset_name = args[1]

        # Check if model_name exists
        model_name = None
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        elif len(args) == 3:
            model_name = args[2]

        # Stateflow
        if model_name is not None and dataset_name is not None:
            _finalize_inference(
                dataset_name=dataset_name,
                model_name=model_name,
            )
        elif dataset_name is not None:
            _finalize_dataset(dataset_name=dataset_name)

        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    """Stateflow Evaluate Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        request_info = None
        if "request_info" in kwargs:
            request_info = kwargs["request_info"]
        elif len(args) == 2:
            request_info = args[1]

        if request_info is not None:
            _evaluate_inference(request_info)
            results = fn(*args, **kwargs)
            _evaluate_inference_finished(request_info)
            return results
        else:
            return fn(*args, **kwargs)

    return wrapper


def delete_dataset(fn: callable) -> callable:
    """Stateflow Delete Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        dataset_name = None
        if len(args) == 2:
            dataset_name = args[1]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        else:
            return fn(*args, **kwargs)

        # Block while performing deletion
        _delete_dataset(dataset_name)

        results = fn(*args, **kwargs)

        # Remove references to deleted data
        _dereference_dataset(dataset_name)

        return results

    return wrapper


def delete_model(fn: callable) -> callable:
    """Stateflow Delete Model Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        model_name = None
        if len(args) == 2:
            model_name = args[1]
        elif "model_name" in kwargs:
            model_name = kwargs["model_name"]
        else:
            return fn(*args, **kwargs)

        # Block while performing deletion
        _delete_model(model_name)

        results = fn(*args, **kwargs)

        # Remove references to deleted data
        _dereference_model(model_name)

        return results

    return wrapper
