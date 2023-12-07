from sqlalchemy import Select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models, ops


def _get_labels(
    db: Session,
    stmt: Select,
) -> set[schemas.Label]:
    """Evaluates statement to get labels."""
    return {
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in db.query(stmt).all()
        if label
    }


def _get_label_keys(
    db: Session,
    stmt: Select,
) -> set[str]:
    """Evaluates statement to get label keys."""
    return {label.key for label in db.query(stmt).all() if label}


def get_labels(
    db: Session,
    filters: schemas.Filter | None = None,
) -> set[schemas.Label]:
    """
    Returns a set of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type).

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).any()
    return _get_labels(db, stmt)


def get_groundtruth_labels(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    """
    Returns a set of unique groundtruth labels.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).groundtruths()
    return _get_labels(db, stmt)


def get_prediction_labels(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    """
    Returns a set of unique prediction labels.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).predictions()
    return _get_labels(db, stmt)


def get_label_keys(
    db: Session,
    filters: schemas.Filter | None = None,
) -> set[schemas.Label]:
    """
    Returns all unique label keys.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).any()
    return _get_label_keys(db, stmt)


def get_groundtruth_label_keys(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    """
    Returns all unique groundtruth label keys.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).groundtruths()
    return _get_label_keys(db, stmt)


def get_prediction_label_keys(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    """
    Returns all unique prediction label keys.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = ops.Query(models.Label).filter(filters).predictions()
    return _get_label_keys(db, stmt)


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
    return list(
        get_groundtruth_labels(db, gt_filter).intersection(
            get_prediction_labels(db, pd_filter)
        )
    )


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
    return list(
        get_groundtruth_label_keys(db, gt_filter).intersection(
            get_prediction_label_keys(db, pd_filter)
        )
    )


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
    ds_labels = get_groundtruth_labels(db, gt_filter)
    md_labels = get_prediction_labels(db, pd_filter)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    # returns tuple of label lists
    return (ds_unique, md_unique)


def get_disjoint_keys(
    db: Session, dataset_name: str, model_name: str, task_type: enums.TaskType
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
    ds_keys = get_groundtruth_label_keys(db, gt_filter)
    md_keys = get_prediction_label_keys(db, pd_filter)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_keys - md_keys)
    md_unique = list(md_keys - ds_keys)

    # returns tuple of label lists
    return (ds_unique, md_unique)
