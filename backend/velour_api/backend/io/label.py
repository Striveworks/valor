from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, select

from velour_api import enums, schemas
from velour_api.backend import models, core, query


def get_labels(
    db: Session,
    key: str = None,
    dataset_name: str = None,
    model_name: str = None,
    task_type: list[enums.TaskType] = [],
) -> list[schemas.Label]:

    dataset = core.get_dataset(db, dataset_name) if dataset_name else None
    model = core.get_model(db, model_name) if model_name else None

    labels = query.get_labels(
        db,
        key=key,
        dataset=dataset,
        model=model,
        task_type=task_type,
    )

    return [
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in labels
    ]


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []


def get_scored_label_distribution(
    db: Session,
) -> list[schemas.ScoredLabelDistribution]:
    return []


def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns tuple(gt_labels, pd_labels)"""

    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)

    ds_labels = set(query.get_labels(db, dataset=dataset))
    md_labels = set(query.get_labels(db, dataset=dataset, model=model))

    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    return {
        "dataset": ds_unique,
        "model": md_unique,
    }
