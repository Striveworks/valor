import json

from sqlalchemy.orm import Session
from sqlalchemy import select, text, insert
from geoalchemy2.functions import ST_GeomFromGeoJSON

from velour_api import schemas
from velour_api.backend import models


def create_image_metadatum(
    db: Session,
    image: schemas.ImageMetadatum,
):
    mapping =  {
        "height": image.height,
        "width": image.width,
        "frame": image.frame,
    }

    added_id = db.scalars(
        insert(models.MetaDatum)
        .values(mapping)
        .returning(models.MetaDatum.id)
    )
    db.commit()
    return added_id.one()



def create_metadatum_mapping(
    db: Session,
    metadatum: schemas.MetaDatum,
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    geometry_id: int = None,
) -> dict:

    if not (dataset_id or model_id or datum_id or geometry_id):
        raise ValueError("Need some target to attach metadatum to.")
    
    ret = {
        "name": metadatum.name,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "datum_id": datum_id,
        "geometry_id": geometry_id,
        "string_value": None,
        "numeric_value": None,
        "geo": None,
        "image_id": None,
    }

    if isinstance(metadatum.value, str):
        ret["string_value"] = metadatum.value
    elif isinstance(metadatum.value, float):
        ret["numeric_value"] = metadatum.value
    elif isinstance(metadatum.value, schemas.GeographicFeature):
        ret["geo"] = ST_GeomFromGeoJSON(json.dumps(metadatum.value.geography))
    elif isinstance(metadatum.value, schemas.ImageMetadata):
        ret["image_id"] = create_image_metadatum(metadatum.value)
    else:
        raise ValueError(
            f"Got unexpected value of type '{type(metadatum.value)}' for metadatum"
        )
    return ret


def create_metadata(
    db: Session,
    metadata: list[schemas.MetaDatum],
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):

    mapping = [
        create_metadatum_mapping(metadatum)
        for metadatum in metadata
    ]

    added_ids = db.scalars(
        insert(models.MetaDatum)
        .values(mapping)
        .returning(models.MetaDatum.id)
    )
    db.commit()
    return added_ids.all()


def query_by_metadata(metadata: list[schemas.MetaDatum]):
    """Returns a subquery of ground truth / predictions that meet the criteria."""
    pass
