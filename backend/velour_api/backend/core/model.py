from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.dataset import get_datum
from velour_api.backend.core.annotation import create_annotation
from velour_api.backend.core.label import create_label
from velour_api.backend.core.metadata import create_metadata 


def get_model(
    db: Session,
    name: str,
) -> models.Model:
    return (
        db.query(models.Model)
        .where(models.Model.name == name)
        .one_or_none()
    )


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    model = get_model(db, name=prediction.model_name)
    datum = get_datum(db, prediction.datum.uid)
    
    rows = []
    for pd in prediction.annotations:
        annotation = create_annotation(db, pd.annotation)
        rows += [
            models.Prediction(
                score=scored_label.score,
                datum=datum,
                model=model,
                annotation=annotation,
                label=create_label(db, scored_label.label),
            )
            for scored_label in pd.scored_labels
        ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.PredictionAlreadyExistsError
    return rows


def create_model(
    db: Session,
    model: schemas.Model,
):
    # Check if dataset already exists.
    if (
        db.query(models.Model)
        .where(models.Model.name == model.name)
        .one_or_none()
    ):
        raise exceptions.DatasetAlreadyExistsError(model.name)

    # Create model
    row = models.Model(name=model.name)
    try:
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)

    # Create metadata
    create_metadata(db, model.metadata, model=row)
    return row


def delete_model(
    db: Session,
    name: str,
):
    try:
        db.delete(models.Model).where(models.Model.name == name)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError