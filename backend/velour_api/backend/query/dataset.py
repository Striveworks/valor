from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    
    print(dataset_name, datum_uid)
    
    # Get datum with metadata
    datum = (
        db.query(models.Datum)
        .join(models.MetaDatum, models.Datum.id == models.MetaDatum.datum_id)
        .join(models.Dataset, models.Datum.dataset_id == models.Dataset.id)
        .where(
            and_(
                models.Datum.uid == datum_uid,
                models.Dataset.name == dataset_name,
            )
        )
        .one_or_none()
    )

    # Get annotations with metadata
    annotation_ids = (
        db.query(models.GroundTruth.annotation_id)
        .join(models.Datum, models.Datum.id == models.GroundTruth.datum_id)
        .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
        .where(
            and_(
                models.Datum.uid == datum_uid,
                models.Dataset.name == dataset_name,
            )
        )
        .distinct()
        .subquery()
    )
    annotations = (
        db.query(models.Annotation)
        .join(models.MetaDatum, models.MetaDatum.annotation_id == models.Annotation.id)
        .where(models.Annotation.id.in_(annotation_ids))
        .all()
    )

    # Get groundtruths
    groundtruths = (
        db.query(
            models.GroundTruth.id, 
            models.GroundTruth.datum_id,
            models.GroundTruth.annotation_id,
            models.GroundTruth.label_id,
        )
        .join(models.Datum, models.Datum.id == models.GroundTruth.datum_id)
        .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
        .where(
            and_(
                models.Datum.uid == datum_uid,
                models.Dataset.name == dataset_name,
            )
        )
        .all()
    )

    return schemas.GroundTruth(
        dataset_name=dataset_name,
        datum=schemas.Datum(
            uid=datum_uid,
            metadata=[
                get
            ]
        )
    )


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:

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

    return schemas.Dataset(id=dataset.id, name=dataset.name, metadata=metadata)


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    return [
        get_dataset(db, name)
        for name in db.query(models.Dataset.name).all()
    ]
