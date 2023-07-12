from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models, core
from velour_api.backend.query.metadata import get_metadata


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    
    print(dataset_name, datum_uid)

    # Get dataset
    dataset = core.get_dataset(db, dataset_name)
    
    # Get datum
    datum = core.get_datum(db, datum_uid)

    # Check
    assert datum.dataset_id == dataset.id

    # Get groundtruths
    groundtruths = (
        db.query(models.GroundTruth)
        .where(models.GroundTruth.datum_id == datum.id)
        .all()
    )

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
        dataset_name=dataset_name,
        datum=schemas.Datum(
            uid=datum_uid,
            metadata=get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.Annotation(
                task_type=annotation.task_type,
                metadata=get_metadata(db, annotation=annotation),
                bounding_box=schemas.BoundingBox.from_wkt(annotation.box),
                polygon=schemas.Polygon.from_wkt(annotation.polygon),
                multipolygon=schemas.MultiPolygon.from_wkt(annotation.multipolygon),
                raster=schemas.Raster.from_gdal(annotation.raster),
            )
            for annotation in annotations
        ]
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
