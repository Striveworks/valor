from sqlalchemy import Subquery, and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models, ops


def fetch_label(
    db: Session,
    label: schemas.Label,
) -> models.Label | None:
    """
    Fetch label from the database.

    Parameters
    ----------
    db : Session
        SQLAlchemy ORM session.
    label : schemas.Label
        Label schema to search for in the database.

    Returns
    -------
    models.Label | None
    """
    return db.query(
        select(models.Label)
        .where(
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
        )
        .subquery()
    ).one_or_none()


def fetch_matching_labels(
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
    return db.query(
        select(models.Label).where(
            or_(
                *[
                    and_(
                        models.Label.key == label.key,
                        models.Label.value == label.value,
                    )
                    for label in labels
                ]
            )
        )
        .subquery()
    ).all()


def create_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Add a list of labels to create in the database.

    Handles cases where the label already exists in the database.

    The returned list of `models.Label` retains the inputs ordering.

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
    # remove duplicates
    labels_no_duplicates = list(set(labels))

    # get existing labels
    existing_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, labels_no_duplicates)
    }

    # create new labels
    new_labels_set = {
        (label.key, label.value)
        for label in labels_no_duplicates
        if (label.key, label.value) not in existing_labels
    }
    new_labels = [
        models.Label(key=label[0], value=label[1])
        for label in list(new_labels_set)
    ]

    # upload the labels that were missing
    try:
        db.add_all(new_labels)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e  # this should never be called

    # get existing labels and match output order to users request
    existing_labels = {
        (label.key, label.value): label
        for label in fetch_matching_labels(db, labels_no_duplicates)
    }
    return [existing_labels[(label.key, label.value)] for label in labels]


def _getter_statement(
    selection,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> Subquery:
    """Builds sql statement for other functions."""
    stmt = ops.Query(selection)
    if filters:
        stmt = stmt.filter(filters)
    if not ignore_groundtruths and ignore_predictions:
        stmt = stmt.groundtruths(as_subquery=False)
    elif ignore_groundtruths and not ignore_predictions:
        stmt = stmt.predictions(as_subquery=False)
    else:
        stmt = stmt.any(as_subquery=False)
    return stmt


def get_labels(
    db: Session,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> set[schemas.Label]:
    """
    Returns a set of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type).

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.
    ignore_groundtruths : bool, default=False
        An optional toggle to ignore labels associated with groundtruths.
    ignore_predictions : bool, default=False
        An optional toggle to ignore labels associated with predictions.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = _getter_statement(
        selection=models.Label,
        filters=filters,
        ignore_groundtruths=ignore_groundtruths,
        ignore_predictions=ignore_predictions,
    )
    return {
        schemas.Label(key=label.key, value=label.value)
        for label in db.query(stmt.subquery()).all()
    }


def get_label_keys(
    db: Session,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> set[str]:
    """
    Returns all unique label keys.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.
    ignore_groundtruths : bool, default=False
        An optional toggle to ignore label keys associated with groundtruths.
    ignore_predictions : bool, default=False
        An optional toggle to ignore label keys associated with predictions.

    Returns
    ----------
    set[str]
        A set of label keys.
    """
    stmt = _getter_statement(
        selection=models.Label.key,
        filters=filters,
        ignore_groundtruths=ignore_groundtruths,
        ignore_predictions=ignore_predictions,
    )
    return {key for key in db.scalars(stmt)}


def get_joint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> list[schemas.Label]:
    """
    Returns all unique labels that are shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_types: list[enums.TaskType]
        The task types to filter on.
    groundtruth_type: enums.AnnotationType
        The groundtruth type to filter on.
    prediction_type: enums.AnnotationType
        The prediction type to filter on

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    gt_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=task_types,
        annotation_types=[groundtruth_type],
    )
    pd_filter = schemas.Filter(
        dataset_names=[dataset_name],
        models_names=[model_name],
        task_types=task_types,
        annotation_types=[prediction_type],
    )

    ds_labels = get_labels(db, gt_filter, ignore_predictions=True)
    md_labels = get_labels(db, pd_filter, ignore_groundtruths=True)

    return list(ds_labels.intersection(md_labels))


def get_joint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    task_type: enums.TaskType,
) -> list[schemas.Label]:
    """
    Returns all unique label keys that are shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_type: enums.TaskType
        The task types to filter on.

    Returns
    ----------
    set[schemas.Label]
        A list of labels.
    """
    gt_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=[task_type],
    )
    pd_filter = schemas.Filter(
        dataset_names=[dataset_name],
        models_names=[model_name],
        task_types=[task_type],
    )

    ds_keys = get_label_keys(db, gt_filter, ignore_predictions=True)
    md_keys = get_label_keys(db, pd_filter, ignore_groundtruths=True)

    return list(ds_keys.intersection(md_keys))


def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique labels that are not shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_types: list[enums.TaskType]
        The task types to filter on.
    groundtruth_type: enums.AnnotationType
        The groundtruth type to filter on.
    prediction_type: enums.AnnotationType
        The prediction type to filter on

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint labels, where the first element is those labels which are present in groundtruths but absent in predictions.
    """

    # create filters
    gt_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=task_types,
        annotation_types=[groundtruth_type],
    )
    pd_filter = schemas.Filter(
        dataset_names=[dataset_name],
        models_names=[model_name],
        task_types=task_types,
        annotation_types=[prediction_type],
    )

    # get labels
    ds_labels = get_labels(db, gt_filter, ignore_predictions=True)
    md_labels = get_labels(db, pd_filter, ignore_groundtruths=True)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    # returns tuple of label lists
    return (ds_unique, md_unique)


def get_disjoint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    task_type: enums.TaskType,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique label keys that are not shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_type: task_type: enums.TaskType
        The task type to filter on.

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint label key, where the first element is those labels which are present in groundtruths but absent in predictions.
    """
    # create filters
    gt_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=[task_type],
    )
    pd_filter = schemas.Filter(
        dataset_names=[dataset_name],
        models_names=[model_name],
        task_types=[task_type],
    )

    # get keys
    ds_keys = get_label_keys(db, gt_filter, ignore_predictions=True)
    md_keys = get_label_keys(db, pd_filter, ignore_groundtruths=True)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_keys - md_keys)
    md_unique = list(md_keys - ds_keys)

    # returns tuple of label lists
    return (ds_unique, md_unique)
