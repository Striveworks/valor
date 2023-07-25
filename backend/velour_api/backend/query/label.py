from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models, query, ops


def get_labels(
    db: Session,
    request_info: schemas.Filter | None,
) -> list[schemas.Label]:
    """Returns a list of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    labels = (
       ops.BackendQuery.label()
       .filter(request_info)
       .all(db)
    )

    return [
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in labels
    ]


def get_joint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> list[schemas.Label]:
    
    # create filters
    dsf = schemas.Filter(filter_by_dataset_names=[dataset_name])
    mdf = schemas.Filter(
        filter_by_dataset_names=[dataset_name],
        filter_by_model_names=[model_name],
    )

    # get label sets
    ds = set(get_labels(db, dsf))
    md = set(get_labels(db, mdf))

    # return intersection of label sets
    return list(ds.intersection(md))


def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns tuple with elements (dataset, model) which contain lists of Labels. """

    # create filters
    ds_filter = schemas.Filter(
        filter_by_dataset_names=[dataset_name]
    )
    md_filter = schemas.Filter(
        filter_by_model_names=[model_name],
        filter_by_dataset_names=[dataset_name],
    )

    # get label sets
    ds_labels = set(get_labels(db, ds_filter))
    md_labels = set(get_labels(db, md_filter))

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    # returns tuple of label lists 
    return (ds_unique, md_unique)


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []
