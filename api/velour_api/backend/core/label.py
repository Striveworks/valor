from sqlalchemy import Select, and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models


def create_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Add a list of labels to postgis. Handles cases where the label already exists in the database.

    Parameters
    -------
    db : Session
        The database session to query against.
    labels : list[schemas.Label]
        A list of labels to add to postgis.

    Returns
    -------
    List[models.Label]
        A list of labels.
    """

    # get existing labels
    existing_labels = set(
        (label.key, label.value)
        for label in _get_existing_labels(db=db, labels=labels)
    )

    # determine which labels already exist
    labels_to_be_added_to_db = []
    for label in labels:
        lookup = (label.key, label.value)
        if lookup not in existing_labels:
            new_label = models.Label(key=label.key, value=label.value)
            labels_to_be_added_to_db.append(new_label)
            existing_labels.add(lookup)

    # upload the labels that were missing
    try:
        db.add_all(labels_to_be_added_to_db)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e  # this should never be called
    
    return _get_existing_labels(db=db, labels=labels)


def _get_existing_labels(
    db: Session,
    labels: schemas.Label,
) -> list[models.Label] | None:
    """
    Fetch labels from postgis that match some list of labels (in terms of both their keys and values).
    """
    label_keys, label_values = zip(
        *[(label.key, label.value) for label in labels]
    )
    return (
        db.query(models.Label)
        .where(
            and_(
                models.Label.key.in_(label_keys),
                models.Label.value.in_(label_values),
            )
        )
        .all()
    )


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
