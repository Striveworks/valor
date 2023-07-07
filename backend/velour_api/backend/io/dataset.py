from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


def request_dataset(
    db: Session,
    name: str,
) -> schemas.DatasetInfo:

    dataset = db.query(models.Dataset).where(models.Dataset.name == name).one_or_none()
    if not dataset:
        return None

    metadata = []
    for row in db.query(models.MetaDatum).where(models.MetaDatum.dataset_id == dataset.id).all():
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

    return schemas.DatasetInfo(id=dataset.id, name=dataset.name, metadata=metadata)


def request_datasets(
    db: Session,
) -> list[schemas.DatasetInfo]:
    return [
        request_dataset(db, name)
        for name in db.query(models.Dataset.name).all()
    ]
