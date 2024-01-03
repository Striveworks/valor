import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models
from velour_api.backend.core.annotation import (
    _create_annotation,
    _create_empty_annotation,
    _create_skipped_annotation,
)


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    """
    Creates a groundtruth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth: schemas.GroundTruth
        The groundtruth to create.
    """
    # create datum
    datum = core.create_datum(db, groundtruth.datum)

    # create annotations and labels
    groundtruth_list = []
    annotation_list = []
    label_list = []
    if not groundtruth.annotations:
        annotation_list = [_create_empty_annotation(datum, None)]
    else:
        for annotation in groundtruth.annotations:
            annotation_list.append(_create_annotation(annotation, datum, None))
            label_list = core.create_labels(db, annotation.labels)

    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.GroundTruthAlreadyExistsError

    return rows


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    """
    Fetch a groundtruth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name : str
        The name of the dataset.
    datum_uid: str
        The UID of the datum to fetch.


    Returns
    ----------
    schemas.GroundTruth
        The requested groundtruth.
    """
    # retrieve from table
    dataset = core.fetch_dataset(db, name=dataset_name)
    datum = core.fetch_datum(db, dataset_id=dataset.id, uid=datum_uid)

    geo_dict = (
        schemas.geojson.from_dict(
            json.loads(db.scalar(ST_AsGeoJSON(datum.geo)))
        )
        if datum.geo
        else None
    )

    return schemas.GroundTruth(
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset_name,
            metadata=datum.meta,
            geospatial=geo_dict,
        ),
        annotations=core.get_annotations(db, datum),
    )
