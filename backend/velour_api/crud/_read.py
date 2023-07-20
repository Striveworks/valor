from sqlalchemy.orm import Session

from velour_api import backend, schemas, enums
from velour_api.backend import state


""" Labels """


@state.read
def get_labels(
    db: Session,
    key: str = None,
    dataset_name: str = None,
    model_name: str = None,
):
    """Retrieves all existing labels."""
    return backend.get_labels(
        db,
        key=key,
        dataset_name=dataset_name,
        model_name=model_name,
    )

@state.read
def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns a dictionary containing disjoint sets of labels. Keys are (dataset, model) and contain sets of labels disjoint from the other."""

    return backend.get_disjoint_labels(
        db, 
        dataset_name=dataset_name, 
        model_name=model_name,
    )

@state.read
def get_disjoint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[str]]:
    """Returns a dictionary containing disjoint sets of label keys. Keys are (dataset, model) and contain sets of keys disjoint from the other."""
    
    disjoint_labels = backend.get_disjoint_labels(
        db, 
        dataset_name=dataset_name, 
        model_name=model_name,
    )

    disjoint_keys = {}
    disjoint_keys["dataset"] = list(set(
        label.key
        for label in disjoint_labels["dataset"]
    ))
    disjoint_keys["model"] = list(set(
        label.key
        for label in disjoint_labels["model"]
    ))
    return disjoint_keys


""" Datum """


# @TODO
@state.read
def get_datum(
    db: Session,
    dataset_name: str,
    uid: str,
) -> schemas.Datum | None:
    # Check that uid is associated with dataset
    return None

# @TODO
@state.read
def get_datums(
    db: Session,
    dataset_name: str,
    filter: enums.DataType,
) -> list[schemas.Datum]:
    return []


""" Datasets """


@state.read
def get_datasets(db: Session) -> list[schemas.Dataset]:
    return backend.get_datasets()


@state.read
def get_dataset(db: Session, name: str) -> schemas.Dataset:
    return backend.get_dataset(db, name)


@state.read
def get_groundtruth(
    db: Session, dataset_name: str, datum_uid: str
) -> schemas.GroundTruth:
    return backend.get_groundtruth(
        db, dataset_name=dataset_name, datum_uid=datum_uid
    )


@state.read
def get_dataset_labels(
    db: Session, name: str
) -> list[schemas.LabelDistribution]:
    pass


""" Models """


@state.read
def get_models(db: Session) -> list[schemas.Model]:
    return backend.get_models(db)


@state.read
def get_model(db: Session, name: str) -> schemas.Model:
    return backend.get_model(db, name)


@state.read
def get_prediction(
    db: Session, model_name: str, datum_uid: str
) -> schemas.Prediction:
    return backend.get_prediction(
        db, model_name=model_name, datum_uid=datum_uid
    )


@state.read
def get_model_labels(
    db: Session, name: str
) -> list[schemas.ScoredLabelDistribution]:
    pass


""" Evaluation """

def get_metrics_from_evaluation_settings_id(
    db: Session, evaluation_settings_id: int
) -> list[schemas.Metric]:
    return backend.get_metrics_from_evaluation_settings_id(db, evaluation_settings_id)


def get_confusion_matrices_from_evaluation_settings_id(
    db: Session, evaluation_settings_id: int
) -> list[schemas.ConfusionMatrix]:
    return backend.get_confusion_matrices_from_evaluation_settings_id(db, evaluation_settings_id)


def get_evaluation_settings_from_id(
    db: Session, evaluation_settings_id: int
) -> schemas.EvaluationSettings:
    return backend.get_evaluation_settings_from_id(db, evaluation_settings_id)


def get_model_metrics(
    db: Session, model_name: str, evaluation_settings_id: int
) -> list[schemas.Metric]:
    return backend.get_model_metrics(db, model_name, evaluation_settings_id)


def get_model_evaluation_settings(
    db: Session, model_name: str
) -> list[schemas.EvaluationSettings]:
    return backend.get_model_evaluation_settings(db, model_name)