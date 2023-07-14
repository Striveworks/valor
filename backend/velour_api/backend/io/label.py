from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, query


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
