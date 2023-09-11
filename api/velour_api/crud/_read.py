from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas
from velour_api.backend import jobs


def get_evaluation_status(
    *, dataset_name: str, model_name: str, job_id: int
) -> enums.JobStatus:
    return jobs.get_stateflow().get_job_status(
        dataset_name=dataset_name,
        model_name=model_name,
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


def get_dataset_evaluations(dataset_name: str):
    return jobs.get_stateflow().get_dataset_jobs(dataset_name)


def get_model_evaluations(model_name: str):
    return jobs.get_stateflow().get_model_jobs(model_name)


""" Labels """


def get_labels(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Label]:
    """Retrieves all existing labels that meet the filter request."""
    return backend.get_labels(db, request)


def get_label_distribution(
    *,
    db: Session,
    request: schemas.Filter,
) -> list[schemas.LabelDistribution]:
    return []


def get_joint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns a dictionary containing disjoint sets of labels. Keys are (dataset, model) and contain sets of labels disjoint from the other."""

    return backend.get_joint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
    )


def get_disjoint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """Returns a dictionary containing disjoint sets of labels. Keys are (dataset, model) and contain sets of labels disjoint from the other."""

    return backend.get_disjoint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
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


def get_groundtruths(
    *,
    db: Session,
    request: schemas.Filter,
) -> list[schemas.GroundTruth]:
    return backend.get_groundtruths(db, request)


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
    *, db: Session, model_name: str, datum_uid: str
) -> schemas.Prediction:
    return backend.get_prediction(
        db, model_name=model_name, datum_uid=datum_uid
    )


# @TODO
def get_predictions(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Prediction]:
    return []


""" Evaluation """


def get_metrics_from_evaluation_settings_id(
    *, db: Session, evaluation_settings_id: int
) -> list[schemas.Metric]:
    return backend.get_metrics_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


def get_confusion_matrices_from_evaluation_settings_id(
    *, db: Session, evaluation_settings_id: int
) -> list[schemas.ConfusionMatrix]:
    return backend.get_confusion_matrices_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


def get_evaluation_settings_from_id(
    *, db: Session, evaluation_settings_id: int
) -> schemas.EvaluationSettings:
    return backend.get_evaluation_settings_from_id(db, evaluation_settings_id)


def get_model_metrics(
    *, db: Session, model_name: str, evaluation_settings_id: int
) -> list[schemas.Metric]:
    return backend.get_model_metrics(db, model_name, evaluation_settings_id)


def get_model_evaluation_settings(
    *, db: Session, model_name: str
) -> list[schemas.EvaluationSettings]:
    return backend.get_model_evaluation_settings(db, model_name)
