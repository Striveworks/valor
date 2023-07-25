from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models, query, ops


def get_labels(
    db: Session,
    request_info: schemas.Filter,
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


def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns dictionary with keys (model, dataset) which contain lists of Labels. """

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

    return {
        "dataset": ds_unique,
        "model": md_unique,
    }


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []
