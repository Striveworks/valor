from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


def create_groundtruths(
    db: Session,
    groundtruths: list[schemas.GroundTruth],
    ignore_existing_datums: bool = False,
):
    """Create ground truths in bulk.

    Parameters
    ----------
    db
        The database Session to query against.
    groundtruths
        The ground truths to create.
    ignore_existing_datums
        If True, will ignore datums that already exist in the database.
        If False, will raise an error if any datums already exist.
        Default is False.

    Returns
    -------
    None
    """
    # check status of dataset(s)
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

    # create datums
    datums = core.create_datums(
        db,
        [groundtruth.datum for groundtruth in groundtruths],
        [
            dataset_name_to_dataset[groundtruth.dataset_name]
            for groundtruth in groundtruths
        ],
        ignore_existing_datums=ignore_existing_datums,
    )
    if ignore_existing_datums:
        # datums only contains the newly created ones, so we need to filter out
        # the ones that already existed
        groundtruths = [
            gt
            for gt in groundtruths
            if (dataset_name_to_dataset[gt.dataset_name].id, gt.datum.uid)
            in datums
        ]

    # retrieve datum ids
    datum_ids = [
        datums[(dataset_name_to_dataset[gt.dataset_name].id, gt.datum.uid)]
        for gt in groundtruths
        if (dataset_name_to_dataset[gt.dataset_name].id, gt.datum.uid)
        in datums
    ]

    # create labels
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
        datum_ids=datum_ids,
        models_=None,
    )

    # create groundtruths
    groundtruth_rows = []
    for groundtruth, annotation_ids_per_groundtruth in zip(
        groundtruths, annotation_ids
    ):
        for i, annotation in enumerate(groundtruth.annotations):
            if annotation.labels:
                for label in annotation.labels:
                    groundtruth_rows.append(
                        models.GroundTruth(
                            annotation_id=annotation_ids_per_groundtruth[i],
                            label_id=label_dict[(label.key, label.value)],
                        )
                    )
    try:
        db.add_all(groundtruth_rows)
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
