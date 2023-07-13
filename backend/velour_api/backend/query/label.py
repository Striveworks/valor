from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models, core, subquery


def get_labels(
    db: Session,
    key: str = None,
    dataset_name: str = None,
    model_name: str = None,
) -> list[schemas.Label]:
    
    # Filter by dataset/model and key
    if dataset_name or model_name:
        # Get sql models (if requested)
        dataset = core.get_dataset(db, dataset_name) if dataset_name else None
        model = core.get_dataset(db, dataset_name) if dataset_name else None

        label_subquery = subquery.get_labels(
            db,
            dataset=dataset,
            model=model,
        )

        if key:
            labels = (
                db.query(label_subquery)
                .where(label_subquery.key == key)
                .all()
            )
        else:
            labels = (
                db.query(label_subquery)
                .all()
            )

    # Filter by key only
    elif key:
        labels = (
            db.query(models.Label)
            .where(models.Label.key == key)
            .all()
        )

    # Return all existing labels
    else:
        labels = db.query(models.Label).all()

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