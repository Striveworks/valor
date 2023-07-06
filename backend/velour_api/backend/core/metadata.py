import json

from sqlalchemy.orm import Session
from sqlalchemy import select, text, insert
from geoalchemy2.functions import ST_GeomFromGeoJSON

from velour_api import schemas
from velour_api.backend import models


def create_image_metadatum(
    db: Session,
    image: schemas.ImageMetadata,
) -> models.ImageMetadata:
    image_metadatum_row = models.ImageMetadata(
        height=image.height,
        width=image.width,
        frame=image.frame,
    )
    db.add(image_metadatum_row)
    db.commit()
    return image_metadatum_row


def create_metadatum(
    db: Session,
    metadatum: schemas.MetaDatum,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    geometry: models.GeometricAnnotation = None,
    commit: bool = True,
) -> dict:

    if not (dataset or model or datum or geometry):
        raise ValueError("Need some target to attach metadatum to.")
    
    mapping = {
        "name": metadatum.name,
        "dataset_id": dataset.id if dataset else None,
        "model_id": model.id if model else None,
        "datum_id": datum.id if datum else None,
        "geometry_id": geometry.id if geometry else None,
        "string_value": None,
        "numeric_value": None,
        "geo": None,
        "image_id": None,
    }

    # Check value type
    if isinstance(metadatum.value, str):
        mapping["string_value"] = metadatum.value
    elif isinstance(metadatum.value, float):
        mapping["numeric_value"] = metadatum.value
    elif isinstance(metadatum.value, schemas.GeographicFeature):
        mapping["geo"] = ST_GeomFromGeoJSON(json.dumps(metadatum.value.geography))
    elif isinstance(metadatum.value, schemas.ImageMetadata):
        mapping["image_id"] = create_image_metadatum(metadatum.value).id
    else:
        raise ValueError(
            f"Got unexpected value of type '{type(metadatum.value)}' for metadatum"
        )
    
    row = models.MetaDatum(**mapping)
    if commit:
        db.add(row)
        db.commit()
    return row


def create_metadata(
    db: Session,
    metadata: list[schemas.MetaDatum],
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    geometry: models.GeometricAnnotation = None,
) -> list[models.MetaDatum]:
    rows = [
        create_metadatum(
            db,
            metadatum,
            dataset=dataset,
            model=model,
            datum=datum,
            geometry=geometry,
            commit=False,
        )
        for metadatum in metadata
    ]
    db.add_all(rows)
    db.commit()
    return rows


def query_by_metadata(metadata: list[schemas.MetaDatum]):
    """Returns a subquery of ground truth / predictions that meet the criteria."""
    pass
