from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.dataset import get_datum
from velour_api.backend.core.annotation import create_annotation
from velour_api.backend.core.label import create_label
from velour_api.backend.core.metadata import create_metadata 


def create_predictions(
    db: Session,
    annotated_datums: list[schemas.Prediction],
    model: models.Model,
):
    rows = []
    for annotated_datum in annotated_datums:
        datum = get_datum(db, annotated_datum.datum)
        for pd in annotated_datum.pds:
            geometry = (
                create_geometric_annotation(db, pd.geometry)
                if pd.geometry
                else None
            )
            rows += [
                models.GroundTruth(
                    datum_id=datum.id,
                    model_id=model.id,
                    task_type=pd.task_type,
                    geometry=geometry.id,
                    label=create_label(db, scored_label.label).id,
                    score=scored_label.score,
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