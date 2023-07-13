import json

from geoalchemy2.functions import ST_GeomFromGeoJSON
from sqlalchemy import insert, select, text, and_
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