from sqlalchemy import Select, and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models


def _get_existing_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Fetch matching labels from the database.

    If a label in the search set is not found, no output is generated.

    Parameters
    ----------
    db : Session
        SQLAlchemy ORM session.
    labels : List[schemas.Label]
        List of label schemas to search for in the database.
    """
    label_keys, label_values = zip(
        *[(label.key, label.value) for label in labels]
    )
    existing_label_kv_combos = {
        (label.key, label.value) : label
        for label in (
            db.query(models.Label)
            .where(
                and_(
                    models.Label.key.in_(label_keys),
                    models.Label.value.in_(label_values),
                )
            )
            .all()
        )
    }
    return [
        existing_label_kv_combos[(label.key, label.value)]
        for label in labels
        if (label.key, label.value) in existing_label_kv_combos
    ]


def create_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Add a list of labels to create in the database. 
    
    Handles cases where the label already exists in the database.

    Parameters
    -------
    db : Session
        The database session to query against.
    labels : list[schemas.Label]
        A list of labels to add to postgis.

    Returns
    -------
    List[models.Label]
        A list of corresponding label rows from the database.
    """
    
    # get existing labels
    label_keys, label_values = zip(
        *[(label.key, label.value) for label in labels]
    )
    existing_labels = {
        (label.key, label.value) : label
        for label in (
            db.query(models.Label)
            .where(
                and_(
                    models.Label.key.in_(label_keys),
                    models.Label.value.in_(label_values),
                )
            )
            .all()
        )
    }

    # create new labels
    new_labels = {
        (label.key, label.value) : models.Label(key=label.key, value=label.value)
        for label in labels
        if (label.key, label.value) not in existing_labels
    }

    # upload the labels that were missing
    try:
        db.add_all(list(new_labels.values()))
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e # this should never be called
    
    # get existing labels
    return _get_existing_labels(db, labels)


def get_label(
    db: Session,
    label: schemas.Label,
) -> models.Label | None:
    """
    Fetch a label from the database.

    Parameters
    -------
    db : Session
        The database session to query against.
    label : schemas.Label
        The label to fetch.

    Returns
    -------
    models.Label
        The requested label.
    """
    return (
        db.query(models.Label)
        .where(
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
        )
        .one_or_none()
    )


def get_labels(
    db: Session,
    annotation: models.Annotation,
) -> list[models.Annotation]:
    """
    Fetch labels associated with an annotation from the database.

    Parameters
    -------
    db : Session
        The database session to query against.
    annotation : models.Annotation
        The annotation to fetch labels for.

    Returns
    -------
    List[models.Annotation]
        The requested list of labels.
    """
    labels = (
        db.query(models.Label.key, models.Label.value)
        .select_from(models.GroundTruth)
        .join(
            models.Label,
            models.GroundTruth.label_id == models.Label.id,
        )
        .where(models.GroundTruth.annotation_id == annotation.id)
        .all()
    )

    labels_with_score = (
        db.query(models.Label.key, models.Label.value, models.Prediction.score)
        .select_from(models.Prediction)
        .join(
            models.Label,
            models.Prediction.label_id == models.Label.id,
        )
        .where(models.Prediction.annotation_id == annotation.id)
        .all()
    )

    if labels:
        return [
            schemas.Label(key=label[0], value=label[1]) for label in labels
        ]
    elif labels_with_score:
        return [
            schemas.Label(key=label[0], value=label[1], score=label[2])
            for label in labels_with_score
        ]
    else:
        raise ValueError(
            f"no labels found for annotation with id: `{annotation.id}`"
        )


def get_dataset_labels_query(
    dataset_name: str,
    annotation_type: enums.AnnotationType,
    task_types: list[enums.TaskType],
) -> Select:
    """
    Create a query to fetch labels associated with a dataset from the database.

    Parameters
    -------
    dataset_name : str
        The dataset to fetch labels for.
    annotation_type : enums.AnnotationType
        The annotation type of the model.
    task_types : listp[enums.TaskType]
        The task types to filter on.

    Returns
    -------
    Select
        A sqlalchemy query.
    """
    annotation_type_expr = (
        [models.annotation_type_to_geometry[annotation_type].is_not(None)]
        if annotation_type is not enums.AnnotationType.NONE
        else []
    )

    return (
        select(models.Label)
        .join(
            models.GroundTruth,
            models.GroundTruth.label_id == models.Label.id,
        )
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(models.Dataset, models.Dataset.id == models.Dataset.id)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Annotation.task_type.in_(task_types),
                *annotation_type_expr,
            )
        )
        .distinct()
    )
