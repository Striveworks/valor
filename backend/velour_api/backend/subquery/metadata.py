from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models

def get_metadatum(
    db: Session, 
    metadatum: models.MetaDatum = None,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Datum = None,
    name: str = None,
) -> schemas.MetaDatum | None:
    
    if not metadatum and name is not None:
        if dataset:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.dataset_id == dataset.id,
                        models.MetaDatum.name == name,
                    )   
                )
                .one_or_none()
            )
        elif model:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.model_id == model.id,
                        models.MetaDatum.name == name,
                    )   
                )
                .one_or_none()
            )
        elif datum:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.datum_id == datum.id,
                        models.MetaDatum.name == name,
                    )   
                )
                .one_or_none()
            )
        elif annotation:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.annotation_id == annotation.id,
                        models.MetaDatum.name == name,
                    )   
                )
                .one_or_none()
            )
        
    # Sanity check
    if metadatum is None:
        return None

    # Parsing
    if metadatum.string_value is not None:
        value = metadatum.string_value
    elif metadatum.numeric_value is not None:
        value = metadatum.numeric_value
    elif metadatum.geo is not None:
        # @TODO: Add geographic type
        raise NotImplemented
    else:
        return None

    return schemas.MetaDatum(
        name=metadatum.name,
        value=value,
    )


def get_metadata(
    db: Session,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[schemas.MetaDatum]:
    
    # @TODO: Make sure that only one is defined?

    if dataset:
        metadata = (
            db.query(models.MetaDatum)
            .where(models.MetaDatum.dataset_id == dataset.id)
            .all()
        )
    elif model:
        metadata = (
            db.query(models.MetaDatum)
            .where(models.MetaDatum.model_id == model.id)
            .all()
        )
    elif datum:
        metadata = (
            db.query(models.MetaDatum)
            .where(models.MetaDatum.datum_id == datum.id)
            .all()
        )
    elif annotation:
        metadata = (
            db.query(models.MetaDatum)
            .where(models.MetaDatum.annotation_id == annotation.id)
            .all()
        )
    else:
        raise RuntimeError


    return [
        get_metadatum(db, metadatum)
        for metadatum in metadata
    ]
