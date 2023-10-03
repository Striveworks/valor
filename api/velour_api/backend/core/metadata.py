import json

from geoalchemy2.functions import ST_GeomFromGeoJSON
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models

UNION_FUNCTIONS = {"or": or_, "and": and_}


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
        "key": metadatum.key,
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


def get_metadatum_schema(
    metadatum: models.MetaDatum,
) -> schemas.MetaDatum | None:
    if metadatum is None:
        return None

    # Parsing
    if metadatum.string_value is not None:
        value = metadatum.string_value
    elif metadatum.numeric_value is not None:
        value = metadatum.numeric_value
    elif metadatum.geo is not None:
        # @TODO: Add geographic type
        raise NotImplementedError
    else:
        return None

    return schemas.MetaDatum(
        key=metadatum.key,
        value=value,
    )


def get_metadata(
    db: Session,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    key: str = None,
    union_type: str = "or",
) -> list[schemas.MetaDatum]:
    """Returns list of metadatums from a union of sources (dataset, model, datum, annotation) filtered by (key, value_type)."""

    if union_type not in UNION_FUNCTIONS:
        raise ValueError(f"union_type must be one of {UNION_FUNCTIONS.keys()}")

    union_func = UNION_FUNCTIONS[union_type]
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
    if key:
        relationships.append(models.MetaDatum.key == key)

    # Add union of relationships
    if len(relationships) == 1:
        metadata = metadata.where(relationships[0])
    elif relationships:
        metadata = metadata.where(union_func(*relationships))

    return [
        get_metadatum_schema(metadatum)
        for metadatum in db.query(metadata.subquery()).all()
    ]
