import collections
import math
from typing import List, Optional

from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas
from velour_api.backend import jobs


def _is_subset_of_dict(target_values: any, nested_dict: dict) -> bool:
    """Helper function that's called recursively to see if a dict exists within a nested dict"""
    for key, value in target_values.items():
        if key not in nested_dict:
            return False
        if isinstance(value, dict):
            if not _is_subset_of_dict(value, nested_dict[key]):
                return False
        elif value != nested_dict[key]:
            return False
    return True


def _dict_is_subset_of_other_dict(
    target_values: any, nested_dict: dict
) -> bool:
    """Check if a target nested_dict exists in any part of an arbitrarily large nested nested_dict"""
    if isinstance(nested_dict, dict):
        if _is_subset_of_dict(target_values, nested_dict):
            return True
        for value in nested_dict.values():
            if _dict_is_subset_of_other_dict(
                target_values=target_values, nested_dict=value
            ):
                return True
    return False


def get_evaluation_status(job_id: int) -> enums.JobStatus:
    return jobs.get_stateflow().get_job_status(
        job_id=job_id,
    )


def get_backend_state(
    *, dataset_name: str, model_name: str | None = None
) -> enums.State:
    stateflow = jobs.get_stateflow()
    if dataset_name and model_name:
        return stateflow.get_inference_status(
            dataset_name=dataset_name,
            model_name=model_name,
        )
    return stateflow.get_dataset_status(
        dataset_name=dataset_name,
    )


def get_evaluation_jobs_for_dataset(dataset_name: str):
    return jobs.get_stateflow().get_dataset_jobs(dataset_name)


def get_evaluation_jobs_for_model(model_name: str):
    return jobs.get_stateflow().get_model_jobs(model_name)


def get_bulk_evaluations(
    db: Session,
    dataset_names: Optional[List[str]],
    model_names: Optional[List[str]],
) -> list[schemas.Evaluation]:
    """
    Returns all metrics associated with user-supplied dataset and model names

    Parameters
    ----------
    db
        The database session
    dataset_names
        A list of dataset names that we want to return metrics for
    model_names
        A list of model names that we want to return metrics for
    """

    job_set = set()

    # get all relevant job IDs from all of the specified models and datasets
    if dataset_names:
        for dataset in dataset_names:
            dataset_jobs = jobs.get_stateflow().get_dataset_jobs(dataset)
            for model, job_ids in dataset_jobs.items():
                job_set.update(job_ids)

    if model_names:
        for model in model_names:
            model_jobs = jobs.get_stateflow().get_model_jobs(model)
            for dataset, job_ids in model_jobs.items():
                job_set.update(job_ids)

    output = backend.get_metrics_from_evaluation_ids(
        db=db, evaluation_ids=job_set
    )
    return output


def get_ranked_evaluations(
    db: Session,
    dataset_name: str,
    metric: str,
    parameters: dict = None,
    label_keys: list = None,
    rank_from_highest_value_to_lowest_value: bool = True,
):
    """
    Returns all metrics associated with a particular dataset, ranked according to user inputs

    Parameters
    ----------
    dataset_name
        The dataset name for which to fetch metrics for.
    metric
        The metric to use when ranking evaluations (e.g., "mAP")
    parameters
        The metric parameters to filter on when computing the ranking (e.g., {'iou':.5}). Will raise a ValueError if the user supplies a metric which requires more granular parameters.
    label_keys
        The list of label keys to filter on (e.g., ['key1'])
    rank_from_highest_value_to_lowest_value
        A boolean to indicate whether the metric values should be ranked from highest to lowest
    """

    metric_to_input_requirements = {
        # evaluate_classification
        "Precision": ["label_keys"],
        "F1": ["label_keys"],
        "Recall": ["label_keys"],
        "ROCAUC": ["label_keys"],
        "Accuracy": ["label_keys"],
        # evaluate_detection
        "AP": ["iou", "label_keys"],
        "APAveragedOverIOUs": ["label_keys"],
        "mAP": ["iou"],
        "mAPAveragedOverIOUs": [],
        # evaluate_segmentation
        "IOUMetric": ["label_keys"],
        "IOUMetricAveraged": [],
    }

    if metric not in metric_to_input_requirements.keys():
        raise ValueError(
            f"Metric should be one of {metric_to_input_requirements.keys()}"
        )

    requirements_for_selection = metric_to_input_requirements[metric]

    if (parameters and not isinstance(parameters, dict)) or (
        label_keys and not isinstance(label_keys, list)
    ):
        raise ValueError(
            "Inputted parameters should be of type dict, while label_keys should be of type list"
        )

    if not label_keys and label_keys in requirements_for_selection:
        raise ValueError(
            f"label_keys argument is required for metric {metric}."
        )

    if ("iou" in requirements_for_selection) and (
        not parameters or "iou" not in parameters
    ):
        raise ValueError("IOU key is missing from parameter dictionary")

    if not parameters:
        parameters = {}

    if not label_keys:
        label_keys = []

    user_label_filter = schemas.LabelFilter(
        keys=label_keys,
    )

    evaluations = get_bulk_evaluations(
        dataset_names=[dataset_name], db=db, model_names=[]
    )

    model_values = collections.defaultdict(float)
    for evaluation in evaluations:
        if (
            not label_keys
            or evaluation.settings.filters.labels == user_label_filter
        ):
            for evaluation_metric in evaluation.metrics:
                if evaluation_metric.type == metric and (
                    not parameters
                    or _dict_is_subset_of_other_dict(
                        target_values=parameters,
                        nested_dict=evaluation_metric.parameters,
                    )
                ):
                    # we should only find one metric per model
                    if evaluation.model in model_values:
                        raise ValueError(
                            "Found multiple metrics per model with the specified label_keys and parameters"
                        )
                    else:
                        model_values[
                            evaluation.model
                        ] = evaluation_metric.value

    rankings = {
        key: rank
        for rank, key in enumerate(
            sorted(
                model_values,
                key=model_values.get,
                reverse=rank_from_highest_value_to_lowest_value,
            ),
            1,
        )
    }

    if not rankings:
        arg_summary = {
            "metric": metric,
            "label_keys": label_keys,
            "parameters": parameters,
        }
        raise ValueError(
            f"Didn't find any evaluations to rank on using {arg_summary}"
        )

    # sort and return evaluations according to this ranking
    for evaluation in evaluations:
        evaluation.ranking = rankings.get(evaluation.model, "not_ranked")

    evaluations = sorted(
        evaluations,
        key=lambda evaluation: rankings.get(evaluation.model, math.inf),
    )
    return evaluations


""" Labels """


def get_all_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Label]:
    """Retrieves all existing labels."""
    return list(backend.get_labels(db, filters))


def get_dataset_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
):
    """Retrieve all labels associated with dataset groundtruths."""
    return list(
        backend.get_groundtruth_labels(
            db=db,
            filters=filters,
        )
    )


def get_model_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
):
    """Retrieve all labels associated with dataset groundtruths."""
    return list(
        backend.get_prediction_labels(
            db=db,
            filters=filters,
        )
    )


def get_joint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> dict[str, list[schemas.Label]]:
    """Returns a dictionary containing disjoint sets of labels. Keys are (dataset, model) and contain sets of labels disjoint from the other."""

    return backend.get_joint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=task_types,
        groundtruth_type=groundtruth_type,
        prediction_type=prediction_type,
    )


def get_disjoint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """Returns a dictionary containing disjoint sets of labels. Keys are (dataset, model) and contain sets of labels disjoint from the other."""

    return backend.get_disjoint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=task_types,
        groundtruth_type=groundtruth_type,
        prediction_type=prediction_type,
    )


def get_disjoint_keys(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_type: enums.TaskType,
) -> tuple[list[str], list[str]]:
    """Returns a dictionary containing disjoint sets of label keys. Keys are (dataset, model) and contain sets of keys disjoint from the other."""

    return backend.get_disjoint_keys(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
    )


""" Datum """


# @TODO
def get_datum(
    *,
    db: Session,
    dataset_name: str,
    uid: str,
) -> schemas.Datum | None:
    # Check that uid is associated with dataset
    return None


# @TODO
def get_datums(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Datum]:
    return backend.get_datums(db, request)


""" Datasets """


def get_dataset(*, db: Session, dataset_name: str) -> schemas.Dataset:
    return backend.get_dataset(db, dataset_name)


def get_datasets(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Dataset]:
    return backend.get_datasets(db)


def get_groundtruth(
    *,
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    return backend.get_groundtruth(
        db,
        dataset_name=dataset_name,
        datum_uid=datum_uid,
    )


""" Models """


def get_model(*, db: Session, model_name: str) -> schemas.Model:
    return backend.get_model(db, model_name)


def get_models(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Model]:
    return backend.get_models(db)


def get_prediction(
    *, db: Session, model_name: str, dataset_name: str, datum_uid: str
) -> schemas.Prediction:
    return backend.get_prediction(
        db,
        model_name=model_name,
        dataset_name=dataset_name,
        datum_uid=datum_uid,
    )


""" Evaluation """


def get_metrics_from_evaluation_ids(
    *, db: Session, evaluation_id: int
) -> list[schemas.Evaluation]:
    return backend.get_metrics_from_evaluation_ids(db, evaluation_id)


def get_evaluation_job_from_id(
    *, db: Session, evaluation_id: int
) -> schemas.EvaluationJob:
    return backend.get_evaluation_job_from_id(db, evaluation_id)


def get_model_metrics(
    *, db: Session, model_name: str, evaluation_id: int
) -> list[schemas.Metric]:
    return backend.get_model_metrics(db, model_name, evaluation_id)


def get_model_evaluation_settings(
    *, db: Session, model_name: str
) -> list[schemas.EvaluationJob]:
    return backend.get_model_evaluation_jobs(db, model_name)
