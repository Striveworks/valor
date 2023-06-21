from functools import wraps

from velour_api.enums import TableStatus
from velour_api.exceptions import (
    DatasetDoesNotExistError,
    InferenceDoesNotExistError,
    InvalidStateError,
)
from velour_api.jobs import get_status, set_status
from velour_api.schemas import (
    APRequest,
    ClfMetricsRequest,
    Dataset,
    DatasetStatus,
    GroundTruthClassificationsCreate,
    GroundTruthDetectionsCreate,
    GroundTruthSegmentationsCreate,
    PredictedClassificationsCreate,
    PredictedDetectionsCreate,
    PredictedSegmentationsCreate,
)


def get_dataset_status(dataset_name: str):
    status = get_status()
    if dataset_name not in status:
        raise DatasetDoesNotExistError(dataset_name)
    elif not status.datasets[dataset_name].valid():
        raise InvalidStateError(f"{dataset_name} is in a invalid state.")
    return status.datasets[dataset_name].status


def get_inference_status(dataset_name: str, model_name: str):
    status = get_status()
    if dataset_name not in status.datasets:
        raise DatasetDoesNotExistError(dataset_name)
    elif not status.datasets[dataset_name].valid():
        raise InvalidStateError(f"{dataset_name} is in a invalid state.")
    elif model_name not in status.datasets[dataset_name].models:
        raise InferenceDoesNotExistError(dataset_name, model_name)
    return status.datasets[dataset_name].models[model_name]


def set_dataset_status(dataset_name: str, state: TableStatus):
    status = get_status()

    # Check if dataset needs to be created
    if dataset_name not in status.datasets:

        # Check if new state is for creation
        if state != TableStatus.CREATE:
            raise InvalidStateError(
                f"New dataset ({dataset_name}) given with state ({state})"
            )

        # Create new dataset
        status.datasets[dataset_name] = DatasetStatus(
            status=state,
            models={},
        )

    # Dataset exists
    else:

        # Check if new state is valid
        if state not in status.datasets[dataset_name].status.next():
            raise InvalidStateError(
                f"{state} is not a valid next state. Valid next states: {status.datasets[dataset_name].status}."
            )

        # Check if inferences (if they exist) allow the next state
        if state not in status.datasets[dataset_name].next():
            raise InvalidStateError(
                f"{state} is not a valid next state. Valid next states: {status.datasets[dataset_name].next()}."
            )

        # Update the state
        status.datasets[dataset_name].status = state

    set_status(status)


def set_inference_status(
    dataset_name: str, model_name: str, state: TableStatus
):
    status = get_status()

    # Check if dataset exists
    if dataset_name not in status.datasets:
        raise DatasetDoesNotExistError(dataset_name)

    # Check if dataset is in evaluation state
    if status.datasets[dataset_name].status != TableStatus.EVALUATE:
        raise InvalidStateError(
            f"dataset {dataset_name} is not in evaluation state."
        )

    # Check if inference does not exist
    if model_name not in status.datasets[dataset_name].models:

        # Check that the state is for creation
        if state != TableStatus.CREATE:
            raise InvalidStateError(
                f"New model ({model_name}) given with state ({state})"
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
            raise InvalidStateError(f"{state} is not a valid next state.")

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
            raise InvalidStateError(
                "Attempted to delete dataset that was not in 'DELETE' state."
            )
    else:
        raise DatasetDoesNotExistError(dataset_name)


def remove_inference(dataset_name: str, model_name: str):
    status = get_status()
    if dataset_name in status.datasets:
        if model_name in status.datasets[dataset_name].models:
            if (
                status.datasets[dataset_name].models[model_name]
                == TableStatus.DELETE
            ):
                del status.datasets[dataset_name].models[model_name]
                set_status(status)
            else:
                raise InvalidStateError(
                    f"Attempted to delete inference ({dataset_name}, {model_name}) that was not in 'DELETE' state."
                )
        if TableStatus.READY in status.datasets[dataset_name].next():
            set_dataset_status(dataset_name, TableStatus.READY)


def remove_model(model_name: str):
    status = get_status()
    for dataset_name in status.datasets:
        if model_name in status.datasets[dataset_name].models:
            remove_inference(dataset_name=dataset_name, model_name=model_name)


def _create_dataset(dataset: Dataset):
    assert isinstance(dataset, Dataset)
    set_dataset_status(dataset_name=dataset.name, state=TableStatus.CREATE)


def _create_data(data):
    if isinstance(
        data,
        (
            GroundTruthClassificationsCreate,
            GroundTruthDetectionsCreate,
            GroundTruthSegmentationsCreate,
        ),
    ):
        set_dataset_status(
            dataset_name=data.dataset_name, state=TableStatus.CREATE
        )
    elif isinstance(
        data,
        (
            PredictedClassificationsCreate,
            PredictedDetectionsCreate,
            PredictedSegmentationsCreate,
        ),
    ):
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


def _finalize_dataset(dataset_name: str):
    assert isinstance(dataset_name, str)
    set_dataset_status(dataset_name=dataset_name, state=TableStatus.READY)


def _evaluate_inference(data):
    assert isinstance(data, (APRequest, ClfMetricsRequest))
    set_inference_status(
        model_name=data.settings.model_name,
        dataset_name=data.settings.dataset_name,
        state=TableStatus.EVALUATE,
    )


def _evaluate_inference_finished(data):
    assert isinstance(data, (APRequest, ClfMetricsRequest))
    set_inference_status(
        model_name=data.settings.model_name,
        dataset_name=data.settings.dataset_name,
        state=TableStatus.READY,
    )


def _delete_inference(model_name: str, dataset_name: str):
    set_inference_status(
        model_name=model_name,
        dataset_name=dataset_name,
        state=TableStatus.DELETE,
    )


def _delete_dataset(dataset_name: str):
    set_dataset_status(dataset_name=dataset_name, state=TableStatus.DELETE)


def _delete_model(model_name: str):
    status = get_status()
    for dataset_name in status.datasets:
        if model_name in status.datasets[dataset_name].models:
            _delete_inference(dataset_name=dataset_name, model_name=model_name)


def _dereference_dataset(dataset_name: str):
    assert isinstance(dataset_name, str)
    remove_dataset(dataset_name)


def _dereference_inference(dataset_name: str, model_name: str):
    assert isinstance(dataset_name, str)
    assert isinstance(model_name, str)
    remove_inference(dataset_name=dataset_name, model_name=model_name)


def _dereference_model(model_name: str):
    assert isinstance(model_name, str)
    remove_model(model_name=model_name)


def create(fn: callable) -> callable:
    """Stateflow Create Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "dataset" in kwargs:
            _create_dataset(kwargs["dataset"])
        elif "model" in kwargs:
            pass
        elif "data" in kwargs:
            _create_data(kwargs["data"])
        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    """Stateflow Finalize Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "model_name" in kwargs and "dataset_name" in kwargs:
            _finalize_inference(
                dataset_name=kwargs["dataset_name"],
                model_name=kwargs["model_name"],
            )
        elif "dataset_name" in kwargs:
            _finalize_dataset(kwargs["dataset_name"])
        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    """Stateflow Evaluate Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "data" in kwargs:
            _evaluate_inference(kwargs["data"])
            print("Starting evaluation")
            results = fn(*args, **kwargs)
            _evaluate_inference_finished(kwargs["data"])
            print("Finished Evaluation")
            return results
        else:
            return fn(*args, **kwargs)

    return wrapper


def delete(fn: callable) -> callable:
    """Stateflow Delete Decorator"""

    @wraps(fn)
    def wrapper(*args, **kwargs):

        # Block while performing deletion
        if "model_name" in kwargs and "dataset_name" in kwargs:
            _delete_inference(
                dataset_name=kwargs["dataset_name"],
                model_name=kwargs["model_name"],
            )
        elif "dataset_name" in kwargs:
            _delete_dataset(kwargs["dataset_name"])
        elif "model_name" in kwargs:
            _delete_model(kwargs["model_name"])

        results = fn(*args, **kwargs)

        # Remove references to deleted data
        if "model_name" in kwargs and "dataset_name" in kwargs:
            _dereference_inference(
                dataset_name=kwargs["dataset_name"],
                model_name=kwargs["model_name"],
            )
        elif "dataset_name" in kwargs:
            _dereference_dataset(kwargs["dataset_name"])
        elif "model_name" in kwargs:
            _dereference_model(kwargs["model_name"])

        return results

    return wrapper
