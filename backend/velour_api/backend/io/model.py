from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models

def request_model(
    db: Session,
    name: str,
) -> schemas.Model:
    
    model = db.query(models.Model).where(models.Model.name == name).one_or_none()
    if not model:
        return None

    metadata = []
    for row in db.query(models.MetaDatum).where(models.MetaDatum.model_id == model.id).all():
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.string_value
                )
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.numeric_value
                )
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.geo,
                )
            )
        elif row.image_id:
            if image_metadatum := db.query(models.ImageMetadata).where(models.ImageMetadata.id == row.image_id).one_or_none():
                metadata.append(
                    schemas.MetaDatum(
                        name=row.name,
                        value=schemas.ImageMetadata(
                            height=image_metadatum.height,
                            width=image_metadatum.width,
                            frame=image_metadatum.frame,
                        ),
                    )
                )

    return schemas.Model(id=model.id, name=model.name, metadata=metadata)


def request_models(
    db: Session,
) -> list[schemas.Model]:
    return [
        request_model(db, name)
        for name in db.query(models.Model.name).all()
    ]