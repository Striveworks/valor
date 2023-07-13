from sqlalchemy import and_
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models, subquery


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:

    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if not dataset:
        return None

    metadata = []
    for row in (
        db.query(models.MetaDatum)
        .where(models.MetaDatum.dataset_id == dataset.id)
        .all()
    ):
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.string_value)
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.numeric_value)
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.geo,
                )
            )
        elif row.image_id:
            if (
                image_metadatum := db.query(models.ImageMetadata)
                .where(models.ImageMetadata.id == row.image_id)
                .one_or_none()
            ):
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
        get_dataset(db, name) for name in db.query(models.Dataset.name).all()
    ]


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:

    # Get dataset
    dataset = core.get_dataset(db, name=dataset_name)

    # Get datum
    datum = core.get_datum(db, uid=datum_uid)

    # Validity check
    if dataset.id != datum.dataset_id:
        raise ValueError(
            f"Datum '{datum_uid}' does not belong to dataset '{dataset_name}'."
        )

    # Get annotations with metadata
    annotations = (
        db.query(models.Annotation)
        .where(
            and_(
                models.Annotation.datum_id == datum.id,
                models.Annotation.model_id.is_(None),
            )
        )
        .all()
    )

    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=subquery.get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.GroundTruthAnnotation(
                labels=subquery.get_labels(db, annotation=annotation),
                annotation=subquery.get_annotation(
                    db, datum=datum, annotation=annotation
                ),
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

    return [get_groundtruth(db, datum.uid) for datum in datums]
