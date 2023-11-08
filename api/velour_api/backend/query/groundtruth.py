from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
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
):
    # retrieve from table
    dataset = core.get_dataset(db, name=dataset_name)
    datum = core.get_datum(db, dataset_id=dataset.id, uid=datum_uid)

    geo_dict = (
        schemas.GeoJSON.from_wkt(datum.geo).to_dict() if datum.geo else {}
    )

    return schemas.GroundTruth(
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset_name,
            metadata=datum.meta,
            geo_metadata=geo_dict,
        ),
        annotations=core.get_annotations(db, datum),
    )
