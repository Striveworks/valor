from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models, core

from velour_api.backend.query.label import get_labels
from velour_api.backend.subquery.metadata import get_metadata
from velour_api.backend.subquery.annotation import get_annotation


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


def get_groundtruth(
    db: Session,
    datum_uid: str,
) -> schemas.GroundTruth:
    
    # Get datum
    datum = core.get_datum(db, uid=datum_uid)

    # Get dataset
    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.id == datum.dataset_id)
        .one_or_none()
    )

    # Sanity check
    assert dataset is not None

    # Get annotations with metadata
    annotation_ids = (
        db.query(models.GroundTruth.annotation_id)
        .where(models.GroundTruth.datum_id == datum.id)
        .distinct()
        .subquery()
    )
    annotations = (
        db.query(models.Annotation)
        .where(models.Annotation.id.in_(annotation_ids))
        .all()
    )

    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.GroundTruthAnnotation(
                labels=get_labels(db, annotation=annotation),
                annotation=get_annotation(db, datum=datum, annotation=annotation),
            )
            for annotation in annotations
        ],
    )


def get_groundtruths(
    db: Session,
    dataset_name: str = None,
) -> list[schemas.GroundTruth]:

    if dataset_name:
        dataset = core.get_dataset(db, dataset_name)
        datums = (
            db.query(models.Datum)
            .where(models.Datum.dataset_id == dataset.id)
            .all()
        )
    else:
        datums = db.query(models.Datum.all())

    return [
        get_groundtruth(db, datum.uid)
        for datum in datums
    ]
    
