from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


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
        The ground truth to create.
    """
    # check dataset status
    if (
        core.get_dataset_status(db=db, name=groundtruth.datum.dataset_name)
        != enums.TableStatus.CREATING
    ):
        raise exceptions.DatasetFinalizedError(groundtruth.datum.dataset_name)

    # create datum
    datum = core.create_datum(db, groundtruth.datum)

    # create labels
    all_labels = [
        label
        for annotation in groundtruth.annotations
        for label in annotation.labels
    ]
    label_list = core.create_labels(db=db, labels=all_labels)

    # create annotations
    annotation_list = core.create_annotations(
        db=db,
        annotations=groundtruth.annotations,
        datum=datum,
        model=None,
    )

    # create groundtruths
    label_idx = 0
    groundtruth_list = []
    for i, annotation in enumerate(groundtruth.annotations):
        for label in label_list[
            label_idx : label_idx + len(annotation.labels)
        ]:
            groundtruth_list.append(
                models.GroundTruth(
                    annotation_id=annotation_list[i].id,
                    label_id=label.id,
                )
            )
        label_idx += len(annotation.labels)

    try:
        db.add_all(groundtruth_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise Exception


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
    return schemas.GroundTruth(
        datum=schemas.Datum(
            uid=datum.uid,
            dataset_name=dataset_name,
            metadata=datum.meta,
        ),
        annotations=core.get_annotations(db, datum),
    )


def delete_groundtruths(
    db: Session,
    dataset: models.Dataset,
):
    """
    Delete all groundtruths from a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : models.Dataset
        The dataset row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if dataset.status != enums.TableStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete groundtruths from dataset `{dataset.name}` which has status `{dataset.status}`"
        )

    subquery = (
        select(models.GroundTruth.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(models.Datum.dataset_id == dataset.id)
        .subquery()
    )
    delete_stmt = delete(models.GroundTruth).where(
        models.GroundTruth.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
