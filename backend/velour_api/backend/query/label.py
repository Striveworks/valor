from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, subquery


def get_labels(
    db: Session,
    key: str = None,
    dataset_name: str = None,
    model_name: str = None,
) -> list[schemas.Label]:

    dataset = core.get_dataset(db, dataset_name) if dataset_name else None
    model = core.get_model(db, model_name) if model_name else None

    return subquery.get_labels(
        db,
        key=key,
        dataset=dataset,
        model=model,
    )


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []


def get_scored_label_distribution(
    db: Session,
) -> list[schemas.ScoredLabelDistribution]:
    return []
