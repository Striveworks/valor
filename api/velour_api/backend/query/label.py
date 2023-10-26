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
    """Returns a list of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""
    stmt = ops.Query(models.Label).filter(filters).any()
    return _get_labels(db, stmt)


def get_groundtruth_labels(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    stmt = ops.Query(models.Label).filter(filters).groundtruths()
    return _get_labels(db, stmt)


def get_prediction_labels(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    stmt = ops.Query(models.Label).filter(filters).predictions()
    return _get_labels(db, stmt)


def get_label_keys(
    db: Session,
    filters: schemas.Filter | None = None,
) -> set[schemas.Label]:
    stmt = ops.Query(models.Label).filter(filters).any()
    return _get_label_keys(db, stmt)


def get_groundtruth_label_keys(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
    stmt = ops.Query(models.Label).filter(filters).groundtruths()
    return _get_label_keys(db, stmt)


def get_prediction_label_keys(
    db: Session,
    filters: schemas.Filter | None,
) -> set[schemas.Label]:
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
    gt_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        annotations=schemas.AnnotationFilter(
            task_types=task_types,
            annotation_types=[groundtruth_type],
        ),
    )
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        models=schemas.ModelFilter(names=[model_name]),
        annotations=schemas.AnnotationFilter(
            task_types=task_types,
            annotation_types=[prediction_type],
        ),
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
    gt_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        annotations=schemas.AnnotationFilter(
            task_types=[task_type],
        ),
    )
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        models=schemas.ModelFilter(names=[model_name]),
        annotations=schemas.AnnotationFilter(
            task_types=[task_type],
        ),
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
) -> dict[str, list[schemas.Label]]:
    """Returns tuple with elements (dataset, model) which contain lists of Labels."""

    # create filters
    gt_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        annotations=schemas.AnnotationFilter(
            task_types=task_types,
            annotation_types=[groundtruth_type],
        ),
    )
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        models=schemas.ModelFilter(names=[model_name]),
        annotations=schemas.AnnotationFilter(
            task_types=task_types,
            annotation_types=[prediction_type],
        ),
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
) -> dict[str, list[schemas.Label]]:
    """Returns tuple with elements (dataset, model) which contain lists of Labels."""

    # create filters
    gt_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        annotations=schemas.AnnotationFilter(
            task_types=[task_type],
        ),
    )
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dataset_name]),
        models=schemas.ModelFilter(names=[model_name]),
        annotations=schemas.AnnotationFilter(
            task_types=[task_type],
        ),
    )

    # get keys
    ds_keys = get_groundtruth_label_keys(db, gt_filter)
    md_keys = get_prediction_label_keys(db, pd_filter)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_keys - md_keys)
    md_unique = list(md_keys - ds_keys)

    # returns tuple of label lists
    return (ds_unique, md_unique)
