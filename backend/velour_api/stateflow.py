from typing import Optional

import schemas
from enums import TableStatus
from jobs import (
    get_dataset_status,
    get_inference_status,
    set_dataset_status,
    set_inference_status,
)


def _flow(current: Optional[TableStatus], next: Optional[TableStatus]) -> bool:
    if next == current:
        allowed = True
    elif next == TableStatus.DELETING:
        allowed = current != TableStatus.EVALUATING
    elif next == TableStatus.EVALUATING:
        allowed = current == TableStatus.READY
    elif next == TableStatus.READY:
        allowed = (
            current == TableStatus.CREATING
            or current == TableStatus.EVALUATING
        )
    elif next == TableStatus.CREATING:
        allowed = current is None
    elif next is None:
        allowed = current == TableStatus.DELETING
    else:
        allowed = False

    if not allowed:
        raise ValueError


def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
            assert isinstance(dataset, schemas.Dataset)
            set_dataset_status(
                dataset_name=dataset.name, status=TableStatus.CREATING
            )
        elif "model" in kwargs:
            pass
            # model = kwargs["model"]
            # assert isinstance(model, schemas.Model)
            # set_inference_status(model_name=model.name, dataset_name=model.da, status=TableStatus.CREATING)
        elif "data" in kwargs:
            data = kwargs["data"]
            if isinstance(
                data,
                (
                    schemas.GroundTruthClassificationsCreate,
                    schemas.GroundTruthDetectionsCreate,
                    schemas.GroundTruthSegmentationsCreate,
                ),
            ):
                dataset_name = data.dataset_name
                current = get_dataset_status(dataset_name=dataset_name)
                if _flow(current, TableStatus.CREATING):
                    set_dataset_status(
                        dataset_name=dataset_name, status=TableStatus.CREATING
                    )
            elif isinstance(
                data,
                (
                    schemas.PredictedClassificationsCreate,
                    schemas.PredictedDetectionsCreate,
                    schemas.PredictedSegmentationsCreate,
                ),
            ):
                dataset_name = data.dataset_name
                model_name = data.model_name
                current = get_inference_status(
                    model_name=model_name, dataset_name=dataset_name
                )
                if _flow(current, TableStatus.CREATING):
                    set_inference_status(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        status=TableStatus.CREATING,
                    )
        else:
            # Do nothing
            pass
        return fn(*args, **kwargs)

    return wrapper


def finalize(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        if "model_name" in kwargs and "dataset_name" in kwargs:
            model_name = kwargs["model_name"]
            dataset_name = kwargs["dataset_name"]
            assert isinstance(model_name, str)
            assert isinstance(dataset_name, str)

            current = get_inference_status(
                model_name=model_name, dataset_name=dataset_name
            )
            if _flow(current, TableStatus.READY):
                set_inference_status(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    status=TableStatus.READY,
                )
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
            assert isinstance(dataset_name, str)
            set_dataset_status(
                dataset_name=dataset_name, status=TableStatus.READY
            )
        else:
            # Do nothing
            pass
        return fn(*args, **kwargs)

    return wrapper


def evaluate(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        if "data" in kwargs:
            data = kwargs["data"]
            assert isinstance(
                data, (schemas.APRequest, schemas.ClfMetricsRequest)
            )
            model_name = data.settings.model_name
            dataset_name = data.settings.dataset_name

            current = get_inference_status(
                model_name=model_name, dataset_name=dataset_name
            )
            if _flow(current, TableStatus.EVALUATING):
                set_inference_status(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    status=TableStatus.EVALUATING,
                )
        else:
            # Do nothing
            pass
        return fn(*args, **kwargs)

    return wrapper


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        if "model_name" in kwargs and "dataset_name" in kwargs:
            model_name = kwargs["model_name"]
            dataset_name = kwargs["dataset_name"]
            current = get_inference_status(
                model_name=model_name, dataset_name=dataset_name
            )
            if _flow(current, TableStatus.DELETING):
                set_inference_status(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    status=TableStatus.DELETING,
                )
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
            current = get_dataset_status(dataset_name=dataset_name)
            if _flow(current, TableStatus.DELETING):
                set_dataset_status(
                    dataset_name=dataset_name, status=TableStatus.DELETING
                )
        return fn(*args, **kwargs)

    return wrapper
