from sqlalchemy import and_
from sqlalchemy.orm import Session

from velour_api import schemas, exceptions
from velour_api.backend import models
from velour_api.backend.core.dataset import get_datum
from velour_api.backend.core.metadata import create_metadata
from velour_api.backend.core.geometry import create_geometric_annotation
from velour_api.backend.core.label import create_label


def create_model_info(db: Session, info: schemas.ModelInfo) -> models.Model:

    # Create model
    row = models.Model(name=info.name)
    db.add(row)
    db.commit()

    # Create metadata
    create_metadata(db, info.metadata, model=row)
    return row


def create_predictions(
    db: Session,
    pds: list[schemas.Prediction],
    model: models.Model,
    datum: models.Datum,
) -> list(models.Prediction):
    rows = []
    for pd in pds:
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
                score=scored_label.score
            )
            for scored_label in pd.scored_labels
        ]
    db.add_all(rows)
    db.commit()
    return rows


def create_model(
    db: Session,
    model: schemas.Model,
):
    # Check if dataset already exists.
    if (
        not db.query(models.Model)
        .where(models.Model.name == model.info.name)
        .one_or_none()
    ):
        raise exceptions.DatasetAlreadyExistsError(model.info.name)

    model_row = create_model_info(db, model.info)
    for datum in model.datums:
        datum_row = get_datum(db, datum)
        create_predictions(
            db, 
            datum.pds, 
            model=model_row, 
            datum=datum_row,
        )
