from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


def create_groundtruths(db: Session, groundtruths: list[schemas.GroundTruth]):
    """Create ground truths in bulk.

    Parameters
    ----------
    db
        The database Session to query against.
    groundtruths
        The ground truths to create.

    Returns
    -------
    None
    """
    # check dataset statuses
    dataset_names = set(
        [groundtruth.dataset_name for groundtruth in groundtruths]
    )
    dataset_name_to_dataset = {
        dataset_name: core.fetch_dataset(db=db, name=dataset_name)
        for dataset_name in dataset_names
    }
    for dataset in dataset_name_to_dataset.values():
        if dataset.status != enums.TableStatus.CREATING:
            raise exceptions.DatasetFinalizedError(dataset.name)

    datums = core.create_datums(
        db,
        [groundtruth.datum for groundtruth in groundtruths],
        [
            dataset_name_to_dataset[groundtruth.dataset_name]
            for groundtruth in groundtruths
        ],
    )
    all_labels = [
        label
        for groundtruth in groundtruths
        for annotation in groundtruth.annotations
        for label in annotation.labels
    ]
    label_dict = core.create_labels(db=db, labels=all_labels)

    # create annotations
    annotation_ids = core.create_annotations(
        db=db,
        annotations=[groundtruth.annotations for groundtruth in groundtruths],
        datums=datums,
        models_=None,
    )

    groundtruth_mappings = []
    for groundtruth, annotation_ids_per_groundtruth in zip(
        groundtruths, annotation_ids
    ):
        for i, annotation in enumerate(groundtruth.annotations):
            for label in annotation.labels:
                groundtruth_mappings.append(
                    {
                        "annotation_id": annotation_ids_per_groundtruth[i],
                        "label_id": label_dict[(label.key, label.value)],
                    }
                )

    try:
        db.bulk_insert_mappings(models.GroundTruth, groundtruth_mappings)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    """
    Fetch a ground truth.

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
        The requested ground truth.
    """
    # retrieve from table
    dataset = core.fetch_dataset(db, name=dataset_name)
    datum = core.fetch_datum(db, dataset_id=dataset.id, uid=datum_uid)
    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
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
