from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models, core

def get_metadatum(db: Session, metadatum: models.MetaDatum):

    if metadatum.string_value:
        value = metadatum.string_value
    elif metadatum.numeric_value:
        value = metadatum.numeric_value
    elif metadatum.geo:
        # @TODO
        raise NotImplemented
        # value = metadatum.geo
    elif metadatum.image:
        image_metadata = (
            db.query(models.ImageMetadata)
            .where(models.ImageMetadata.id == metadatum.image_id)
            .one_or_none()
        )
        value = schemas.ImageMetadata(
            height=image_metadata.height,
            width=image_metadata.width,
            frame=image_metadata.frame,
        )
    else:
        raise ValueError

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

