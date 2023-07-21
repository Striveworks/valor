import json

from geoalchemy2.functions import ST_GeomFromGeoJSON
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models


def create_metadatum(
    db: Session,
    metadatum: schemas.MetaDatum,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    commit: bool = True,
) -> models.MetaDatum:

    if not (dataset or model or datum or annotation):
        raise ValueError("Need some target to attach metadatum to.")

    mapping = {
        "name": metadatum.name,
        "dataset": dataset if dataset else None,
        "model": model if model else None,
        "datum": datum if datum else None,
        "annotation": annotation if annotation else None,
        "string_value": None,
        "numeric_value": None,
        "geo": None,
    }

    # Check value type
    if isinstance(metadatum.value, str):
        mapping["string_value"] = metadatum.value
    elif isinstance(metadatum.value, float):
        mapping["numeric_value"] = metadatum.value
    elif isinstance(metadatum.value, schemas.GeographicFeature):
        mapping["geo"] = ST_GeomFromGeoJSON(
            json.dumps(metadatum.value.geography)
        )
    else:
        raise ValueError(
            f"Got unexpected value of type '{type(metadatum.value)}' for metadatum"
        )

    row = models.MetaDatum(**mapping)
    if commit:
        try:
            db.add(row)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise exceptions.MetaDatumAlreadyExistsError
    return row


def create_metadata(
    db: Session,
    metadata: list[schemas.MetaDatum],
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[models.MetaDatum]:
    if not metadata:
        return None
    rows = [
        create_metadatum(
            db,
            metadatum,
            dataset=dataset,
            model=model,
            datum=datum,
            annotation=annotation,
            commit=False,
        )
        for metadatum in metadata
    ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.MetaDatumAlreadyExistsError
    return rows


def get_metadata(
    db: Session,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    name: str = None,
    value_type: type = None,
) -> list[models.MetaDatum]:
    """Returns list of metadatums from a union of sources (dataset, model, datum, annotation) filtered by (name, value_type)."""

    metadata = select(models.MetaDatum)

    # Query relationships
    relationships = []
    if dataset:
        relationships.append(models.MetaDatum.dataset_id == dataset.id)
    if model:
        relationships.append(models.MetaDatum.model_id == model.id)
    if datum:
        relationships.append(models.MetaDatum.datum_id == datum.id)
    if annotation:
        relationships.append(models.MetaDatum.annotation_id == annotation.id)

    # Add union of relationships
    if len(relationships) == 1:
        metadata = metadata.where(relationships[0])
    elif relationships:
        metadata = metadata.where(or_(*relationships))

    # Filter by name
    if name:
        metadata.where(models.MetaDatum.name == name)

    # Filter by value type
    if value_type:
        if value_type == str:
            metadata.where(models.MetaDatum.string_value.isnot(None))
        elif value_type == float:
            metadata.where(models.MetaDatum.numeric_value.isnot(None))
        else:
            raise NotImplementedError(
                f"Type {str(value_type)} is not currently supported as a metadatum value type."
            )

    return db.query(metadata.subquery()).all()