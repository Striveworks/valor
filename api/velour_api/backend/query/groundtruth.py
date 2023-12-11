import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


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

    annotation_list, label_list = core.create_annotations_and_labels(
        db=db, annotations=groundtruth.annotations, datum=datum
    )

    rows = []

    for i, annotation in enumerate(annotation_list):
        for label in label_list[i]:
            rows += [
                models.GroundTruth(
                    annotation_id=annotation.id, label_id=label.id
                )
            ]

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
    dataset = core.get_dataset(db, name=dataset_name)
    datum = core.get_datum(db, dataset_id=dataset.id, uid=datum_uid)

    geo_dict = (
        json.loads(db.scalar(ST_AsGeoJSON(datum.geo))) if datum.geo else {}
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
