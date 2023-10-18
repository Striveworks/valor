from velour_api import schemas


def deserialize_meta(
    metadata: list[schemas.Metadatum],
) -> dict[str, any]:
    def unpack(metadatum: schemas.Metadatum):
        if isinstance(metadatum.value, schemas.DateTime):
            raise NotImplementedError("Have not implemented DateTime meta")
        return metadatum.value

UNION_FUNCTIONS = {"or": or_, "and": and_}


def create_metadatum(
    db: Session,
    metadatum: schemas.MetaDatum,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
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


def create_metadata_for_multiple_annotations(
    db: Session,
    metadata: list[schemas.MetaDatum],
    annotations: list[models.Annotation],
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
) -> list[models.MetaDatum]:
    """
    Add metadata for multiple annotations to postgis

    Parameters
    -------
    db
        The database session to query against.
    metadata
        The list of metadata you want to upload.
    annotations
        The list of annotations you want to create metadata for.
    dataset
        (Optional) The dataset you want to query against.
    model
        (Optional) The model you want to query against.
    datum
        (Optional) The datum you want to query against.
    """
    if not metadata:
        return None

    rows = []

    for i, annotation in enumerate(annotations):
        if metadata[i]:
            rows.append(
                create_metadatum(
                    db,
                    metadata[i],
                    dataset=dataset,
                    model=model,
                    datum=datum,
                    annotation=annotation,
                )
            )

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
#     return {
#         metadatum.key: unpack(metadatum)
#         for metadatum in metadata
#         if metadatum.value is not None
#     }


def serialize_meta(metadata: dict[str, any]) -> list[schemas.Metadatum]:
    return (
        [schemas.Metadatum(key=key, value=metadata[key]) for key in metadata]
        if metadata
        else []
    )
